/**
 * Iron Man helmet augmentation.
 *
 * Fits a stylized red-and-gold helmet to the tracked face using a handful of
 * stable MediaPipe face-mesh landmarks (forehead, chin, cheekbones, eyes, nose
 * and mouth). The fit is smoothed across frames so the helmet does not jitter
 * with the raw landmarks, and it honors the selfie mirroring via
 * `context.toCanvas`.
 *
 * Rendering is a layered "material" pipeline drawn with the 2D canvas API in a
 * local, roll-aligned space (translate to the midpoint of the eyes, rotate by
 * the eye-line angle):
 *
 *   1. cast shadow  — soft shadow that seats the shell on the underlying face
 *   2. red shell    — banded metallic gradient, yaw-swept dome sheen, vignette
 *   3. ear pods     — concentric gold discs with offset specular caps
 *   4. gold plate   — machined gradient, yaw-swept specular, embossed seams
 *   5. rim light    — additive highlight hugging the shell silhouette
 *   6. eye slits    — angular emissive slits with layered glow + bloom
 *   7. mouth grill  — recessed vent with lit slot lips
 *   8. finishing    — chin vent, nose glint, crisp outline re-strokes
 *
 * Head yaw (the nose-tip offset in the rotated frame) sweeps the specular
 * highlights so the metal visibly responds to head turns. A short appear ramp
 * fades the helmet in when a face is acquired, and a ghost fade masks
 * single-frame detection drop-outs.
 *
 * Performance: every `Path2D` shape and geometry-derived gradient is cached
 * per face and rebuilt only when the quantized fit changes — with EMA-smoothed
 * fits that is a near-100 % hit rate, so a steady helmet costs no path
 * building at all. The only per-frame allocations are the handful of small
 * pulse-driven eye gradients.
 */
import type { AugmentationModule, ModuleContext } from './AugmentationModule';
import type { DetectionResult, Point2D, Point3D } from '../app/types';

/** Face-mesh indices the helmet is anchored to. */
export const IRON_MAN_FACE_INDICES = {
  foreheadTop: 10,
  chin: 152,
  cheekRight: 234,
  cheekLeft: 454,
  eyeOuterRight: 33,
  eyeOuterLeft: 263,
  noseTip: 1,
  mouthUpper: 13,
  mouthLower: 14,
  mouthCornerRight: 61,
  mouthCornerLeft: 291,
} as const;

/** Below this many canvas pixels a face is too small to fit a helmet to. */
const MIN_FACE_PX = 24;

/** How far the head-yaw estimate may swing (lighting is clamped beyond it). */
const YAW_LIMIT = 0.85;

/** Appear / ghost-fade timings (seconds). */
const APPEAR_T = 0.24;
const APPEAR_T_REDUCED = 0.12;
const GHOST_T = 0.2;
const MAX_GHOSTS = 4;

const TAU = Math.PI * 2;

/** Immutable per-frame fit of the helmet, in canvas pixel space. */
export interface IronManFit {
  /** Midpoint of the outer eye corners. */
  center: Point2D;
  /** Eye-line angle in radians, normalized to [-π/2, π/2]. */
  roll: number;
  /** Cheekbone-to-cheekbone width. */
  faceWidth: number;
  /** Forehead-to-chin height. */
  faceHeight: number;
  /** Outer-eye-corner distance (drives the eye slits). */
  eyeDistance: number;
  /** Local y of the forehead landmark relative to `center` (negative = above). */
  browY: number;
  /** Local y of the chin landmark relative to `center` (positive = below). */
  chinY: number;
  /** Local y of the mouth center. */
  mouthY: number;
  /** Corner-to-corner mouth width. */
  mouthWidth: number;
  /** Local y of the nose tip. */
  noseY: number;
  /**
   * Head-turn estimate in [-YAW_LIMIT, YAW_LIMIT]: the nose-tip offset in the
   * roll-aligned frame divided by the half face width. Positive = nose toward
   * local +x. Drives the asymmetric (yaw-aware) lighting; `0` when the nose
   * landmark is missing or unusable.
   */
  yaw: number;
}

function distance(a: Point2D, b: Point2D): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

/** Fall back when an optional landmark produced a non-finite value. */
function finiteOr(value: number, fallback: number): number {
  return Number.isFinite(value) ? value : fallback;
}

function easeOutCubic(t: number): number {
  return 1 - (1 - t) ** 3;
}

/**
 * Estimate the helmet fit for one face. Pure apart from the supplied
 * `toCanvas` converter, which keeps it testable without a DOM.
 *
 * Returns `null` when the required landmarks are missing, non-finite or the
 * face is too small to draw a legible helmet.
 */
export function estimateIronManFit(
  face: readonly Point3D[],
  toCanvas: (point: Point3D) => Point2D,
): IronManFit | null {
  const forehead = face[IRON_MAN_FACE_INDICES.foreheadTop];
  const chin = face[IRON_MAN_FACE_INDICES.chin];
  const cheekRight = face[IRON_MAN_FACE_INDICES.cheekRight];
  const cheekLeft = face[IRON_MAN_FACE_INDICES.cheekLeft];
  const eyeRight = face[IRON_MAN_FACE_INDICES.eyeOuterRight];
  const eyeLeft = face[IRON_MAN_FACE_INDICES.eyeOuterLeft];
  if (!forehead || !chin || !cheekRight || !cheekLeft || !eyeRight || !eyeLeft) {
    return null;
  }

  const pForehead = toCanvas(forehead);
  const pChin = toCanvas(chin);
  const pCheekRight = toCanvas(cheekRight);
  const pCheekLeft = toCanvas(cheekLeft);
  const pEyeRight = toCanvas(eyeRight);
  const pEyeLeft = toCanvas(eyeLeft);

  const center: Point2D = {
    x: (pEyeRight.x + pEyeLeft.x) / 2,
    y: (pEyeRight.y + pEyeLeft.y) / 2,
  };

  // Eye-line angle, folded into [-π/2, π/2]: the helmet is left/right
  // symmetric, so a 180° flip would wrongly turn it upside down.
  let roll = Math.atan2(
    pEyeLeft.y - pEyeRight.y,
    pEyeLeft.x - pEyeRight.x,
  );
  while (roll > Math.PI / 2) roll -= Math.PI;
  while (roll < -Math.PI / 2) roll += Math.PI;

  const faceWidth = distance(pCheekRight, pCheekLeft);
  const faceHeight = distance(pForehead, pChin);
  const eyeDistance = distance(pEyeRight, pEyeLeft);
  if (
    !Number.isFinite(faceWidth) ||
    !Number.isFinite(faceHeight) ||
    !Number.isFinite(eyeDistance) ||
    faceWidth < MIN_FACE_PX ||
    faceHeight < MIN_FACE_PX ||
    eyeDistance < MIN_FACE_PX / 3
  ) {
    return null;
  }

  // Rotate each landmark into the helmet's local frame to find the anchors.
  const cos = Math.cos(roll);
  const sin = Math.sin(roll);
  const localX = (p: Point2D): number => {
    const dx = p.x - center.x;
    const dy = p.y - center.y;
    return dx * cos + dy * sin;
  };
  const localY = (p: Point2D): number => {
    const dx = p.x - center.x;
    const dy = p.y - center.y;
    return -dx * sin + dy * cos;
  };

  const browY = localY(pForehead);
  const chinY = localY(pChin);

  const mouthUpper = face[IRON_MAN_FACE_INDICES.mouthUpper];
  const mouthLower = face[IRON_MAN_FACE_INDICES.mouthLower];
  const mouthY = finiteOr(
    mouthUpper && mouthLower
      ? (localY(toCanvas(mouthUpper)) + localY(toCanvas(mouthLower))) / 2
      : chinY * 0.58,
    chinY * 0.58,
  );

  const mouthCornerRight = face[IRON_MAN_FACE_INDICES.mouthCornerRight];
  const mouthCornerLeft = face[IRON_MAN_FACE_INDICES.mouthCornerLeft];
  const mouthWidth = finiteOr(
    mouthCornerRight && mouthCornerLeft
      ? distance(toCanvas(mouthCornerRight), toCanvas(mouthCornerLeft))
      : faceWidth * 0.3,
    faceWidth * 0.3,
  );

  const noseTip = face[IRON_MAN_FACE_INDICES.noseTip];
  const nosePoint = noseTip ? toCanvas(noseTip) : null;
  const noseY = finiteOr(
    nosePoint ? localY(nosePoint) : chinY * 0.35,
    chinY * 0.35,
  );
  const noseLocalX = nosePoint ? localX(nosePoint) : 0;
  const yaw = Number.isFinite(noseLocalX)
    ? clamp(noseLocalX / (faceWidth * 0.5), -YAW_LIMIT, YAW_LIMIT)
    : 0;

  return {
    center,
    roll,
    faceWidth,
    faceHeight,
    eyeDistance,
    browY,
    chinY,
    mouthY,
    mouthWidth,
    noseY,
    yaw,
  };
}

/** Derived, per-frame geometry shared by every paint layer. */
interface HelmetMetrics {
  halfWidth: number;
  top: number;
  height: number;
  /** Base stroke weight, scaled with the helmet. */
  edge: number;
  /** Inter-pupillary distance clamped to the plate width. */
  iod: number;
  halfMouth: number;
  mouthY: number;
  noseY: number;
  yaw: number;
  /** Maps a 0–1 helmet-space t to local y (0 = crown, 1 = chin). */
  ty(t: number): number;
}

/**
 * Single-slot paint cache per face: paths and gradients are rebuilt only when
 * the quantized geometry key changes. `CanvasGradient` and `Path2D` are not
 * bound to a context, so they are safe to reuse across frames.
 */
class HelmetPaint {
  private readonly gradients = new Map<string, CanvasGradient>();
  private readonly paths = new Map<string, Path2D>();
  private key = '';

  /** Invalidate the cache when the quantized geometry moved. */
  sync(key: string): void {
    if (key !== this.key) {
      this.gradients.clear();
      this.paths.clear();
      this.key = key;
    }
  }

  gradient(
    id: string,
    ctx: CanvasRenderingContext2D,
    build: (ctx: CanvasRenderingContext2D) => CanvasGradient,
  ): CanvasGradient {
    let gradient = this.gradients.get(id);
    if (!gradient) {
      gradient = build(ctx);
      this.gradients.set(id, gradient);
    }
    return gradient;
  }

  path(id: string, build: () => Path2D): Path2D {
    let path = this.paths.get(id);
    if (!path) {
      path = build();
      this.paths.set(id, path);
    }
    return path;
  }
}

interface HelmetState extends IronManFit {
  initialized: boolean;
  /** 0→1 entrance ramp. */
  appear: number;
  /** Geometry-keyed paint cache for this face. */
  paint: HelmetPaint;
}

/** A helmet whose face left the frame, fading out to mask drop-outs. */
interface GhostHelmet {
  fit: IronManFit;
  paint: HelmetPaint;
  alpha: number;
  /** Frozen clock so the glow does not animate while fading. */
  time: number;
}

/** Exponential smoothing time constants (seconds). Lower = snappier. */
const SMOOTH_TAU = {
  center: 0.05,
  roll: 0.07,
  size: 0.09,
  feature: 0.07,
  yaw: 0.1,
} as const;

export class IronManMask implements AugmentationModule {
  id = 'iron-man-mask';
  name = 'Iron Man mask';
  description = 'Red-and-gold Iron Man helmet with metallic shading and glowing eyes.';
  enabled = true;

  private readonly states: Array<HelmetState | undefined> = [];
  private readonly ghosts: GhostHelmet[] = [];
  private readonly reducedMotion: boolean;

  constructor(options: { enabled?: boolean } = {}) {
    this.enabled = options.enabled ?? true;
    let reduced = false;
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }
    this.reducedMotion = reduced;
    if (reduced) {
      this.description = 'Iron Man helmet — eye glow pulse disabled (reduced motion).';
    }
  }

  update(
    ctx: CanvasRenderingContext2D,
    result: DetectionResult,
    context: ModuleContext,
  ): void {
    const dt = clamp(context.dt, 0, 0.1); // tab switches produce huge gaps
    this.drawGhosts(ctx, dt);

    const faces = result.face?.landmarks;
    if (!faces || faces.length === 0) {
      this.retireAll(context.time);
      return;
    }

    // Faces that left the frame become fading ghosts instead of vanishing.
    for (let i = faces.length; i < this.states.length; i++) {
      if (this.states[i]) this.retire(i, context.time);
    }
    if (this.states.length > faces.length) this.states.length = faces.length;

    const appearRate = dt / (this.reducedMotion ? APPEAR_T_REDUCED : APPEAR_T);

    for (let i = 0; i < faces.length; i++) {
      const face = faces[i];
      if (!face) continue;

      const target = estimateIronManFit(face, context.toCanvas);
      if (!target) {
        // Unfittable mid-session (face at the frame edge): fade, don't blink.
        if (this.states[i]) this.retire(i, context.time);
        continue;
      }

      const state = this.smooth(i, target, dt);
      if (state.appear < 1) {
        state.appear = Math.min(1, state.appear + appearRate);
      }

      const ease = easeOutCubic(state.appear);
      const scale = this.reducedMotion ? 1 : 0.94 + 0.06 * ease;

      ctx.save();
      try {
        ctx.translate(state.center.x, state.center.y);
        ctx.rotate(state.roll);
        ctx.scale(scale, scale);
        ctx.globalAlpha *= ease;
        this.paintHelmet(ctx, state.paint, state, context.time);
      } finally {
        ctx.restore(); // never leak transform / shadows to the next module
      }
    }
  }

  onEnable(): void {
    this.reset();
  }

  onDisable(): void {
    this.reset();
  }

  dispose(): void {
    this.reset();
  }

  // -----------------------------------------------------------------------
  // fitting
  // -----------------------------------------------------------------------

  /** EMA-smooth a per-face fit so helmet edges do not shimmer frame to frame. */
  private smooth(index: number, target: IronManFit, dt: number): HelmetState {
    const previous = this.states[index];

    if (!previous || !previous.initialized) {
      const fresh: HelmetState = {
        ...target,
        initialized: true,
        appear: 0,
        paint: new HelmetPaint(),
      };
      this.states[index] = fresh;
      return fresh;
    }

    const alpha = (tau: number): number =>
      dt <= 0 ? 1 : 1 - Math.exp(-dt / tau);

    const aCenter = alpha(SMOOTH_TAU.center);
    previous.center.x += (target.center.x - previous.center.x) * aCenter;
    previous.center.y += (target.center.y - previous.center.y) * aCenter;

    // Fold the roll delta into [-π/2, π/2] so ±90° tilts do not spin around.
    let deltaRoll = target.roll - previous.roll;
    while (deltaRoll > Math.PI / 2) deltaRoll -= Math.PI;
    while (deltaRoll < -Math.PI / 2) deltaRoll += Math.PI;
    previous.roll += deltaRoll * alpha(SMOOTH_TAU.roll);

    const aSize = alpha(SMOOTH_TAU.size);
    previous.faceWidth += (target.faceWidth - previous.faceWidth) * aSize;
    previous.faceHeight += (target.faceHeight - previous.faceHeight) * aSize;
    previous.eyeDistance +=
      (target.eyeDistance - previous.eyeDistance) * aSize;

    const aFeature = alpha(SMOOTH_TAU.feature);
    previous.browY += (target.browY - previous.browY) * aFeature;
    previous.chinY += (target.chinY - previous.chinY) * aFeature;
    previous.mouthY += (target.mouthY - previous.mouthY) * aFeature;
    previous.mouthWidth += (target.mouthWidth - previous.mouthWidth) * aFeature;
    previous.noseY += (target.noseY - previous.noseY) * aFeature;
    previous.yaw += (target.yaw - previous.yaw) * alpha(SMOOTH_TAU.yaw);

    return previous;
  }

  // -----------------------------------------------------------------------
  // lifecycle of per-face states and ghosts
  // -----------------------------------------------------------------------

  private retire(index: number, time: number): void {
    const state = this.states[index];
    if (!state || !state.initialized) return;
    if (this.ghosts.length >= MAX_GHOSTS) this.ghosts.shift();
    this.ghosts.push({
      fit: state,
      paint: state.paint,
      alpha: easeOutCubic(state.appear),
      time,
    });
    this.states[index] = undefined;
  }

  private retireAll(time: number): void {
    for (let i = 0; i < this.states.length; i++) this.retire(i, time);
    this.states.length = 0;
  }

  private reset(): void {
    this.states.length = 0;
    this.ghosts.length = 0;
  }

  private drawGhosts(ctx: CanvasRenderingContext2D, dt: number): void {
    if (this.ghosts.length === 0) return;
    for (let i = this.ghosts.length - 1; i >= 0; i--) {
      const ghost = this.ghosts[i]!;
      ghost.alpha -= dt / GHOST_T;
      if (ghost.alpha <= 0) {
        this.ghosts.splice(i, 1);
        continue;
      }
      ctx.save();
      try {
        ctx.translate(ghost.fit.center.x, ghost.fit.center.y);
        ctx.rotate(ghost.fit.roll);
        ctx.globalAlpha *= clamp(ghost.alpha, 0, 1);
        this.paintHelmet(ctx, ghost.paint, ghost.fit, ghost.time);
      } finally {
        ctx.restore();
      }
    }
  }

  // -----------------------------------------------------------------------
  // painting
  // -----------------------------------------------------------------------

  private paintHelmet(
    ctx: CanvasRenderingContext2D,
    paint: HelmetPaint,
    fit: IronManFit,
    time: number,
  ): void {
    const h = fit.faceHeight;
    const halfWidth = Math.max(fit.faceWidth * 0.62, h * 0.34);
    const top = fit.browY - h * 0.16;
    const bottom = fit.chinY + h * 0.08;
    const height = bottom - top;

    // Reject degenerate / inverted fits instead of drawing garbage.
    if (halfWidth < 6 || height < 8) return;

    const ty = (t: number): number => top + t * height;
    const iod = Math.min(fit.eyeDistance, halfWidth * 1.05);
    const m: HelmetMetrics = {
      halfWidth,
      top,
      height,
      edge: Math.max(1, height * 0.009),
      iod,
      halfMouth: clamp(fit.mouthWidth * 0.62, halfWidth * 0.2, halfWidth * 0.4),
      mouthY: clamp(fit.mouthY, ty(0.68), ty(0.9)),
      noseY: clamp(fit.noseY, ty(0.46), ty(0.62)),
      yaw: fit.yaw,
      ty,
    };

    // Rebuild cached paths / gradients only when the quantized fit moved.
    paint.sync(
      `${Math.round(halfWidth)}:${Math.round(height)}:${Math.round(top)}:` +
        `${Math.round(iod)}:${Math.round(m.halfMouth)}:${Math.round(m.mouthY)}:` +
        `${Math.round(m.noseY)}:${Math.round(fit.yaw * 16)}`,
    );

    const shell = paint.path('shell', () => buildShellPath(halfWidth, top, height));
    const plate = paint.path('plate', () => buildFaceplatePath(halfWidth, top, height));
    const seams = paint.path('seams', () => buildSeamsPath(halfWidth, top, height));
    const centreSeam = paint.path('centreSeam', () =>
      buildCentreSeamPath(top, height),
    );

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    this.drawShell(ctx, paint, m, shell);
    this.drawEarPods(ctx, paint, m);
    this.drawFaceplate(ctx, paint, m, plate, seams, centreSeam);
    this.drawRimLight(ctx, paint, m, shell);

    const pulse = this.reducedMotion
      ? 1
      : 0.92 + 0.08 * Math.sin(time * 2.6);
    this.drawEyes(ctx, paint, m, pulse);

    this.drawGrill(ctx, paint, m);
    this.drawFinishing(ctx, m, plate, shell);
  }

  private drawShell(
    ctx: CanvasRenderingContext2D,
    paint: HelmetPaint,
    m: HelmetMetrics,
    shell: Path2D,
  ): void {
    const shellFill = paint.gradient('shellFill', ctx, (c) =>
      linearGradient(c, 0, m.top, 0, m.top + m.height, SHELL_STOPS),
    );

    // Cast shadow: seats the helmet onto the underlying face.
    ctx.save();
    ctx.shadowColor = 'rgba(12, 2, 4, 0.55)';
    ctx.shadowBlur = m.height * 0.06;
    ctx.shadowOffsetY = m.height * 0.02;
    ctx.shadowOffsetX = m.yaw * m.height * 0.035;
    ctx.fillStyle = shellFill;
    ctx.fill(shell);
    ctx.restore();

    // Dome sheen (swept by head yaw) + edge vignette, clipped to the shell.
    ctx.save();
    ctx.clip(shell);
    ctx.fillStyle = paint.gradient('shellSheen', ctx, (c) => {
      const sx = -m.yaw * m.halfWidth * 0.42 - m.halfWidth * 0.32;
      return radialGradient(c, sx, m.ty(0.07), m.halfWidth * 1.45, SHEEN_STOPS);
    });
    ctx.fillRect(-m.halfWidth * 1.2, m.top, m.halfWidth * 2.4, m.height);
    ctx.fillStyle = paint.gradient('shellVignette', ctx, (c) =>
      radialGradient(c, 0, m.ty(0.5), m.halfWidth * 1.05, VIGNETTE_STOPS),
    );
    ctx.fillRect(-m.halfWidth * 1.2, m.top, m.halfWidth * 2.4, m.height);

    // Machined crest ridge down the crown.
    const crest = paint.path('crest', () => {
      const p = new Path2D();
      p.moveTo(0, m.ty(0.008));
      p.lineTo(0, m.ty(0.1));
      return p;
    });
    ctx.lineWidth = Math.max(1, m.edge * 0.5);
    ctx.save();
    ctx.translate(m.edge * 0.55, 0);
    ctx.strokeStyle = 'rgba(50, 6, 10, 0.85)';
    ctx.stroke(crest);
    ctx.restore();
    ctx.save();
    ctx.translate(-m.edge * 0.4, 0);
    ctx.strokeStyle = 'rgba(255, 228, 216, 0.6)';
    ctx.stroke(crest);
    ctx.restore();
    ctx.restore();

    ctx.lineWidth = m.edge * 1.6;
    ctx.strokeStyle = '#2b060a';
    ctx.stroke(shell);
  }

  private drawEarPods(
    ctx: CanvasRenderingContext2D,
    paint: HelmetPaint,
    m: HelmetMetrics,
  ): void {
    const cy = m.ty(0.47);
    const r = m.height * 0.052;

    for (const side of [-1, 1] as const) {
      const cx = side * m.halfWidth * 0.87;

      // Bezel ring.
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, TAU);
      ctx.fillStyle = paint.gradient(`podRing${side}`, ctx, (c) => {
        const g = c.createRadialGradient(
          cx - r * 0.3,
          cy - r * 0.35,
          r * 0.1,
          cx,
          cy,
          r,
        );
        g.addColorStop(0, '#ffe9ad');
        g.addColorStop(0.55, '#d3ab5a');
        g.addColorStop(1, '#77571d');
        return g;
      });
      ctx.fill();
      ctx.lineWidth = m.edge * 1.2;
      ctx.strokeStyle = 'rgba(40, 22, 6, 0.9)';
      ctx.stroke();

      // Recess.
      ctx.beginPath();
      ctx.arc(cx, cy, r * 0.66, 0, TAU);
      ctx.fillStyle = '#291109';
      ctx.fill();

      // Specular cap.
      ctx.beginPath();
      ctx.arc(cx, cy, r * 0.34, 0, TAU);
      ctx.fillStyle = paint.gradient(`podCap${side}`, ctx, (c) => {
        const g = c.createRadialGradient(
          cx - r * 0.12,
          cy - r * 0.18,
          0,
          cx,
          cy,
          r * 0.42,
        );
        g.addColorStop(0, '#ffefc0');
        g.addColorStop(1, '#8a6a2a');
        return g;
      });
      ctx.fill();
    }
  }

  private drawFaceplate(
    ctx: CanvasRenderingContext2D,
    paint: HelmetPaint,
    m: HelmetMetrics,
    plate: Path2D,
    seams: Path2D,
    centreSeam: Path2D,
  ): void {
    ctx.fillStyle = paint.gradient('plateFill', ctx, (c) =>
      linearGradient(c, 0, m.ty(0.06), 0, m.ty(1), GOLD_STOPS),
    );
    ctx.fill(plate);
    ctx.lineWidth = m.edge * 1.5;
    ctx.strokeStyle = '#3a2208';
    ctx.stroke(plate);

    ctx.save();
    ctx.clip(plate);

    // Yaw-swept specular streak.
    ctx.fillStyle = paint.gradient('plateSpec', ctx, (c) => {
      const sx = -m.yaw * m.halfWidth * 0.5;
      return linearGradient(
        c,
        sx - m.halfWidth * 0.9,
        m.ty(0.12),
        sx + m.halfWidth * 0.6,
        m.ty(0.94),
        PLATE_SPEC_STOPS,
      );
    });
    ctx.fillRect(-m.halfWidth, m.top, m.halfWidth * 2, m.height);

    // Jaw ambient occlusion.
    ctx.fillStyle = paint.gradient('plateAO', ctx, (c) =>
      linearGradient(c, 0, m.ty(0.78), 0, m.ty(1), PLATE_AO_STOPS),
    );
    ctx.fillRect(-m.halfWidth, m.top, m.halfWidth * 2, m.height);

    // Engraved panel seams: horizontal grooves embossed vertically, the
    // centre seam (vertical) embossed horizontally.
    strokeEngraved(ctx, seams, m.edge * 0.45, Math.max(1, m.edge * 0.6), 'y');
    strokeEngraved(ctx, centreSeam, m.edge * 0.45, Math.max(1, m.edge * 0.6), 'x');

    ctx.restore();
  }

  private drawRimLight(
    ctx: CanvasRenderingContext2D,
    paint: HelmetPaint,
    m: HelmetMetrics,
    shell: Path2D,
  ): void {
    ctx.save();
    ctx.clip(shell);
    ctx.globalCompositeOperation = 'lighter';
    ctx.strokeStyle = paint.gradient('rimLight', ctx, (c) =>
      linearGradient(
        c,
        0,
        m.top,
        0,
        m.ty(0.55),
        [
          [0, 'rgba(255, 228, 214, 0.40)'],
          [1, 'rgba(255, 228, 214, 0)'],
        ],
      ),
    );
    ctx.lineWidth = m.edge * 5;
    ctx.stroke(shell);
    ctx.restore();
  }

  private drawEyes(
    ctx: CanvasRenderingContext2D,
    paint: HelmetPaint,
    m: HelmetMetrics,
    pulse: number,
  ): void {
    for (const side of [-1, 1] as const) {
      const slit = paint.path(`slit${side}`, () => buildSlitPath(side, m.iod));
      const socket = paint.path(`socket${side}`, () =>
        buildSlitPath(side, m.iod, 1.16),
      );
      const cx = side * m.iod * 0.5;
      const cy = -m.iod * 0.02;

      // Recessed socket cavity.
      ctx.fillStyle = '#0a0f16';
      ctx.fill(socket);
      ctx.lineWidth = Math.max(1, m.edge * 1.1);
      ctx.strokeStyle = 'rgba(124, 92, 30, 0.85)';
      ctx.stroke(socket);

      // Emissive core + additive bloom.
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';

      const bloom = ctx.createRadialGradient(cx, cy, 0, cx, cy, m.iod * 0.3);
      bloom.addColorStop(0, `rgba(214, 246, 255, ${0.26 * pulse})`);
      bloom.addColorStop(0.45, `rgba(120, 214, 255, ${0.13 * pulse})`);
      bloom.addColorStop(1, 'rgba(80, 180, 240, 0)');
      ctx.fillStyle = bloom;
      ctx.beginPath();
      ctx.ellipse(cx, cy, m.iod * 0.32, m.iod * 0.2, 0, 0, TAU);
      ctx.fill();

      const core = ctx.createLinearGradient(
        0,
        cy - m.iod * 0.1,
        0,
        cy + m.iod * 0.1,
      );
      core.addColorStop(0, '#aee8ff');
      core.addColorStop(0.42, '#f2fdff');
      core.addColorStop(1, '#b8eeff');
      ctx.fillStyle = core;
      ctx.fill(slit);

      // Hot specular streak inside the slit.
      ctx.fillStyle = `rgba(255, 255, 255, ${0.32 * pulse})`;
      ctx.beginPath();
      ctx.ellipse(
        cx - side * m.iod * 0.02,
        cy - m.iod * 0.012,
        m.iod * 0.16,
        m.iod * 0.017,
        -0.06 * side,
        0,
        TAU,
      );
      ctx.fill();
      ctx.restore();

      ctx.lineWidth = Math.max(0.8, m.edge * 0.8);
      ctx.strokeStyle = `rgba(200, 244, 255, ${0.75 * pulse})`;
      ctx.stroke(slit);
    }

    // Brow shadow above the sockets for contrast against the gold.
    ctx.strokeStyle = 'rgba(56, 32, 6, 0.5)';
    ctx.lineWidth = Math.max(1, m.height * 0.011);
    ctx.beginPath();
    for (const side of [-1, 1] as const) {
      ctx.moveTo(side * m.iod * 0.3, -m.iod * 0.115);
      ctx.quadraticCurveTo(
        side * m.iod * 0.5,
        -m.iod * 0.175,
        side * m.iod * 0.7,
        -m.iod * 0.125,
      );
    }
    ctx.stroke();
  }

  private drawGrill(
    ctx: CanvasRenderingContext2D,
    paint: HelmetPaint,
    m: HelmetMetrics,
  ): void {
    const halfH = m.height * 0.055;
    const grill = paint.path('grill', () =>
      buildGrillPath(m.halfMouth, m.mouthY, halfH),
    );

    ctx.fillStyle = paint.gradient('grillCavity', ctx, (c) =>
      linearGradient(
        c,
        0,
        m.mouthY - halfH,
        0,
        m.mouthY + halfH,
        GRILL_STOPS,
      ),
    );
    ctx.fill(grill);
    ctx.lineWidth = m.edge * 1.2;
    ctx.strokeStyle = 'rgba(52, 30, 8, 0.9)';
    ctx.stroke(grill);

    ctx.save();
    ctx.clip(grill);

    // Inner top shadow.
    ctx.fillStyle = paint.gradient('grillShadow', ctx, (c) =>
      linearGradient(
        c,
        0,
        m.mouthY - halfH,
        0,
        m.mouthY,
        GRILL_SHADOW_STOPS,
      ),
    );
    ctx.fillRect(-m.halfMouth * 1.1, m.mouthY - halfH, m.halfMouth * 2.2, halfH);

    // Lit vent slots.
    const lip = Math.max(1, halfH * 0.16);
    for (let i = 0; i < 3; i++) {
      const y = m.mouthY + (i - 1) * halfH * 0.56;
      const w = m.halfMouth * (0.86 - i * 0.04);
      ctx.fillStyle = '#0b0502';
      ctx.fillRect(-w, y - halfH * 0.16, w * 2, halfH * 0.32);
      ctx.fillStyle = 'rgba(232, 196, 130, 0.45)';
      ctx.fillRect(-w, y - halfH * 0.16, w * 2, lip);
    }
    ctx.restore();

    // Bevel highlight along the top edge of the frame.
    ctx.save();
    ctx.translate(0, -m.edge * 0.5);
    ctx.strokeStyle = 'rgba(255, 232, 176, 0.30)';
    ctx.lineWidth = Math.max(0.8, m.edge * 0.8);
    ctx.stroke(grill);
    ctx.restore();
  }

  private drawFinishing(
    ctx: CanvasRenderingContext2D,
    m: HelmetMetrics,
    plate: Path2D,
    shell: Path2D,
  ): void {
    // Chin vent.
    const vw = m.halfWidth * 0.11;
    const vh = Math.max(4, m.height * 0.026);
    const vy = m.ty(0.95);
    roundRectPath(ctx, -vw, vy - vh / 2, vw * 2, vh, vh / 2);
    ctx.fillStyle = '#150b05';
    ctx.fill();
    ctx.lineWidth = Math.max(0.8, m.edge * 0.7);
    ctx.strokeStyle = 'rgba(196, 156, 78, 0.65)';
    ctx.stroke();
    ctx.fillStyle = 'rgba(255, 232, 176, 0.22)';
    ctx.fillRect(-vw * 0.7, vy - vh * 0.3, vw * 1.4, Math.max(1, vh * 0.16));

    // Nose-ridge glint.
    ctx.beginPath();
    ctx.ellipse(
      -m.halfWidth * 0.045,
      m.noseY,
      m.halfWidth * 0.055,
      m.height * 0.016,
      -0.25,
      0,
      TAU,
    );
    ctx.fillStyle = 'rgba(255, 246, 214, 0.45)';
    ctx.fill();

    // Crisp final outlines over every layer beneath.
    ctx.lineWidth = m.edge * 1.5;
    ctx.strokeStyle = 'rgba(43, 6, 10, 0.9)';
    ctx.stroke(shell);
    ctx.lineWidth = m.edge;
    ctx.strokeStyle = 'rgba(58, 34, 8, 0.85)';
    ctx.stroke(plate);
  }
}

// -------------------------------------------------------------------------
// material stop tables
// -------------------------------------------------------------------------

type GradientStops = ReadonlyArray<readonly [number, string]>;

/** Banded hot-rod red — the broad stops read as machined metal. */
const SHELL_STOPS: GradientStops = [
  [0.0, '#ff8272'],
  [0.06, '#f0554a'],
  [0.16, '#d03328'],
  [0.3, '#b0231e'],
  [0.46, '#9c1a17'],
  [0.6, '#b0231e'],
  [0.78, '#751214'],
  [1.0, '#440a0e'],
];

const SHEEN_STOPS: GradientStops = [
  [0, 'rgba(255, 231, 224, 0.34)'],
  [0.42, 'rgba(255, 176, 166, 0.10)'],
  [0.75, 'rgba(20, 0, 6, 0.05)'],
  [1, 'rgba(20, 0, 6, 0.26)'],
];

const VIGNETTE_STOPS: GradientStops = [
  [0, 'rgba(18, 1, 5, 0)'],
  [0.68, 'rgba(18, 1, 5, 0)'],
  [1, 'rgba(18, 1, 5, 0.3)'],
];

const GOLD_STOPS: GradientStops = [
  [0.0, '#ffedb8'],
  [0.05, '#f9dd94'],
  [0.13, '#f0c973'],
  [0.22, '#e2b155'],
  [0.33, '#c99844'],
  [0.47, '#b98a3c'],
  [0.6, '#a67a31'],
  [0.74, '#8a6322'],
  [0.88, '#6c4c17'],
  [1.0, '#4d340d'],
];

const PLATE_SPEC_STOPS: GradientStops = [
  [0, 'rgba(255, 252, 236, 0)'],
  [0.3, 'rgba(255, 252, 236, 0.30)'],
  [0.46, 'rgba(255, 250, 230, 0.10)'],
  [0.62, 'rgba(255, 244, 210, 0)'],
  [1, 'rgba(46, 26, 6, 0.12)'],
];

const PLATE_AO_STOPS: GradientStops = [
  [0, 'rgba(40, 20, 4, 0)'],
  [1, 'rgba(40, 20, 4, 0.32)'],
];

const GRILL_STOPS: GradientStops = [
  [0, '#1d1007'],
  [0.5, '#0d0704'],
  [1, '#1a0e06'],
];

const GRILL_SHADOW_STOPS: GradientStops = [
  [0, 'rgba(0, 0, 0, 0.5)'],
  [1, 'rgba(0, 0, 0, 0)'],
];

// -------------------------------------------------------------------------
// path / gradient helpers
// -------------------------------------------------------------------------

type Side = -1 | 1;

function linearGradient(
  ctx: CanvasRenderingContext2D,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  stops: GradientStops,
): CanvasGradient {
  const g = ctx.createLinearGradient(x0, y0, x1, y1);
  for (const [offset, color] of stops) g.addColorStop(offset, color);
  return g;
}

function radialGradient(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  radius: number,
  stops: GradientStops,
): CanvasGradient {
  const g = ctx.createRadialGradient(x, y, 0, x, y, radius);
  for (const [offset, color] of stops) g.addColorStop(offset, color);
  return g;
}

/** Emboss a seam: dark stroke on one side, light stroke on the other. */
function strokeEngraved(
  ctx: CanvasRenderingContext2D,
  path: Path2D,
  depth: number,
  width: number,
  axis: 'x' | 'y',
): void {
  const dx = axis === 'x' ? depth : 0;
  const dy = axis === 'y' ? depth : 0;
  ctx.lineWidth = width;
  ctx.save();
  ctx.translate(dx, dy);
  ctx.strokeStyle = 'rgba(58, 32, 6, 0.75)';
  ctx.stroke(path);
  ctx.restore();
  ctx.save();
  ctx.translate(-dx * 0.7, -dy * 0.7);
  ctx.strokeStyle = 'rgba(255, 236, 180, 0.30)';
  ctx.stroke(path);
  ctx.restore();
}

/**
 * Cranium shell: slightly boxier jaw than a plain oval so the silhouette
 * reads "engineered" rather than "balloon".
 */
function buildShellPath(halfWidth: number, top: number, height: number): Path2D {
  const ty = (t: number): number => top + t * height;
  const p = new Path2D();
  p.moveTo(0, ty(0));
  // Right side: crown → temple → jaw → chin.
  p.bezierCurveTo(
    halfWidth * 0.52, ty(-0.012),
    halfWidth * 0.93, ty(0.09),
    halfWidth * 0.985, ty(0.28),
  );
  p.bezierCurveTo(
    halfWidth * 1.0, ty(0.43),
    halfWidth * 0.96, ty(0.56),
    halfWidth * 0.88, ty(0.68),
  );
  p.bezierCurveTo(
    halfWidth * 0.76, ty(0.83),
    halfWidth * 0.52, ty(0.965),
    halfWidth * 0.24, ty(1),
  );
  p.lineTo(-halfWidth * 0.24, ty(1));
  // Mirrored left side.
  p.bezierCurveTo(
    -halfWidth * 0.52, ty(0.965),
    -halfWidth * 0.76, ty(0.83),
    -halfWidth * 0.88, ty(0.68),
  );
  p.bezierCurveTo(
    -halfWidth * 0.96, ty(0.56),
    -halfWidth * 1.0, ty(0.43),
    -halfWidth * 0.985, ty(0.28),
  );
  p.bezierCurveTo(
    -halfWidth * 0.93, ty(0.09),
    -halfWidth * 0.52, ty(-0.012),
    0, ty(0),
  );
  p.closePath();
  return p;
}

/**
 * Gold faceplate: flares past the cheekbones (covering the wearer's cheeks)
 * and tapers into the chin, leaving a red band around the silhouette.
 */
function buildFaceplatePath(
  halfWidth: number,
  top: number,
  height: number,
): Path2D {
  const ty = (t: number): number => top + t * height;
  const p = new Path2D();
  p.moveTo(0, ty(0.13));
  p.bezierCurveTo(
    halfWidth * 0.34, ty(0.105),
    halfWidth * 0.6, ty(0.19),
    halfWidth * 0.74, ty(0.31),
  );
  p.bezierCurveTo(
    halfWidth * 0.86, ty(0.45),
    halfWidth * 0.82, ty(0.62),
    halfWidth * 0.66, ty(0.755),
  );
  p.bezierCurveTo(
    halfWidth * 0.52, ty(0.875),
    halfWidth * 0.34, ty(0.955),
    halfWidth * 0.17, ty(0.985),
  );
  p.lineTo(-halfWidth * 0.17, ty(0.985));
  p.bezierCurveTo(
    -halfWidth * 0.34, ty(0.955),
    -halfWidth * 0.52, ty(0.875),
    -halfWidth * 0.66, ty(0.755),
  );
  p.bezierCurveTo(
    -halfWidth * 0.82, ty(0.62),
    -halfWidth * 0.86, ty(0.45),
    -halfWidth * 0.74, ty(0.31),
  );
  p.bezierCurveTo(
    -halfWidth * 0.6, ty(0.19),
    -halfWidth * 0.34, ty(0.105),
    0, ty(0.13),
  );
  p.closePath();
  return p;
}

/** All engraved panel seams in one path (they share a stroke style). */
function buildSeamsPath(halfWidth: number, top: number, height: number): Path2D {
  const ty = (t: number): number => top + t * height;
  const p = new Path2D();
  // Brow ridge arc.
  p.moveTo(-halfWidth * 0.6, ty(0.355));
  p.quadraticCurveTo(0, ty(0.245), halfWidth * 0.6, ty(0.355));
  for (const side of [-1, 1] as const) {
    // Cheek crease.
    p.moveTo(side * halfWidth * 0.62, ty(0.42));
    p.quadraticCurveTo(
      side * halfWidth * 0.52,
      ty(0.62),
      side * halfWidth * 0.3,
      ty(0.845),
    );
    // Nose ridge.
    p.moveTo(side * halfWidth * 0.075, ty(0.4));
    p.quadraticCurveTo(
      side * halfWidth * 0.062,
      ty(0.5),
      side * halfWidth * 0.05,
      ty(0.575),
    );
  }
  // Nose base.
  p.moveTo(-halfWidth * 0.1, ty(0.6));
  p.quadraticCurveTo(0, ty(0.635), halfWidth * 0.1, ty(0.6));
  // Chin seam.
  p.moveTo(-halfWidth * 0.22, ty(0.93));
  p.quadraticCurveTo(0, ty(0.968), halfWidth * 0.22, ty(0.93));
  return p;
}

/** Vertical centre seam down the forehead (embossed along x). */
function buildCentreSeamPath(top: number, height: number): Path2D {
  const ty = (t: number): number => top + t * height;
  const p = new Path2D();
  p.moveTo(0, ty(0.16));
  p.lineTo(0, ty(0.3));
  return p;
}

/**
 * Key points of one eye slit in the local frame (y = 0 is the eye line).
 * The outer end is raised and cut at an angle — the movie silhouette.
 */
function slitPoints(side: Side, iod: number, grow: number): Array<[number, number]> {
  const xIn = side * iod * 0.315;
  const xOut = side * iod * 0.685;
  const lift = -iod * 0.05;
  const hIn = iod * 0.062;
  const hOut = iod * 0.088;
  const pts: Array<[number, number]> = [
    [xIn, lift * 0.25 + hIn * 0.85], // inner bottom
    [xIn, lift * 0.25 - hIn * 0.7], // inner top
    [xOut, lift - hOut * 0.4], // outer top
    [xOut, lift + hOut * 0.3], // outer bottom
  ];
  if (grow === 1) return pts;
  const cx = (xIn + xOut) / 2;
  const cy = lift * 0.1;
  return pts.map(
    (p): [number, number] => [
      cx + (p[0] - cx) * grow,
      cy + (p[1] - cy) * grow,
    ],
  );
}

function buildSlitPath(side: Side, iod: number, grow = 1): Path2D {
  const [innerBottom, innerTop, outerTop, outerBottom] = slitPoints(side, iod, grow);
  const midX = (innerBottom[0] + outerTop[0]) / 2;
  const p = new Path2D();
  p.moveTo(innerBottom[0], innerBottom[1]);
  p.quadraticCurveTo(
    midX,
    innerBottom[1] + iod * 0.02,
    outerBottom[0],
    outerBottom[1],
  );
  p.lineTo(outerTop[0], outerTop[1]);
  p.quadraticCurveTo(
    midX,
    innerTop[1] - iod * 0.02,
    innerTop[0],
    innerTop[1],
  );
  p.closePath();
  return p;
}

/** Mouth grill: rounded frame tapering slightly toward the chin. */
function buildGrillPath(halfMouth: number, y: number, halfH: number): Path2D {
  const topHalf = halfMouth;
  const bottomHalf = halfMouth * 0.8;
  const r = halfH * 0.5;
  const t = y - halfH;
  const b = y + halfH;
  const p = new Path2D();
  p.moveTo(-topHalf + r, t);
  p.lineTo(topHalf - r, t);
  p.quadraticCurveTo(topHalf, t, topHalf - r * 0.35, t + r);
  p.lineTo(bottomHalf + r * 0.35, b - r);
  p.quadraticCurveTo(bottomHalf, b, bottomHalf - r, b);
  p.lineTo(-bottomHalf + r, b);
  p.quadraticCurveTo(-bottomHalf, b, -bottomHalf - r * 0.35, b - r);
  p.lineTo(-topHalf + r * 0.35, t + r);
  p.quadraticCurveTo(-topHalf, t, -topHalf + r, t);
  p.closePath();
  return p;
}

function roundRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number,
): void {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
