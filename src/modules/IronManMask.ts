/**
 * Iron Man helmet augmentation.
 *
 * Fits a stylized red-and-gold helmet to the tracked face using a handful of
 * stable MediaPipe face-mesh landmarks (forehead, chin, cheekbones, eyes and
 * mouth). The fit is smoothed across frames so the helmet does not jitter with
 * the raw landmarks, and it honors the selfie mirroring via `context.toCanvas`.
 *
 * Everything is drawn with the 2D canvas API in a local, roll-aligned space:
 * translate to the midpoint of the eyes, rotate by the eye line angle, then
 * draw the shell / faceplate / glowing eye slits / mouth grill.
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

  // Rotate each landmark into the helmet's local frame to find vertical
  // anchors. Local x is not needed — the helmet is symmetric.
  const localY = (p: Point2D): number => {
    const dx = p.x - center.x;
    const dy = p.y - center.y;
    return -dx * Math.sin(roll) + dy * Math.cos(roll);
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
  const noseY = finiteOr(
    noseTip ? localY(toCanvas(noseTip)) : chinY * 0.35,
    chinY * 0.35,
  );

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
  };
}

interface SmoothedFit extends IronManFit {
  initialized: boolean;
}

/** Exponential smoothing time constants (seconds). Lower = snappier. */
const SMOOTH_TAU = {
  center: 0.05,
  roll: 0.07,
  size: 0.09,
  feature: 0.07,
} as const;

export class IronManMask implements AugmentationModule {
  id = 'iron-man-mask';
  name = 'Iron Man mask';
  description = 'Augments your face with a red-and-gold Iron Man helmet.';
  enabled = true;

  private readonly states: Array<SmoothedFit | undefined> = [];
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
    const faces = result.face?.landmarks;
    if (!faces || faces.length === 0) {
      this.states.length = 0;
      return;
    }

    // Drop stale per-face smoothing state when faces leave the frame.
    if (this.states.length > faces.length) this.states.length = faces.length;

    const dt = clamp(context.dt, 0, 0.1); // tab switches produce huge gaps
    const time = context.time;

    for (let i = 0; i < faces.length; i++) {
      const face = faces[i];
      if (!face) continue;
      const target = estimateIronManFit(face, context.toCanvas);
      if (!target) continue;

      const fit = this.smooth(i, target, dt);

      ctx.save();
      try {
        ctx.translate(fit.center.x, fit.center.y);
        ctx.rotate(fit.roll);
        this.drawHelmet(ctx, fit, time);
      } finally {
        ctx.restore(); // never leak transform / shadows to the next module
      }
    }
  }

  onEnable(): void {
    this.states.length = 0;
  }

  onDisable(): void {
    this.states.length = 0;
  }

  dispose(): void {
    this.states.length = 0;
  }

  // -----------------------------------------------------------------------
  // fitting
  // -----------------------------------------------------------------------

  /** EMA-smooth a per-face fit so helmet edges do not shimmer frame to frame. */
  private smooth(index: number, target: IronManFit, dt: number): IronManFit {
    const previous = this.states[index];

    if (!previous || !previous.initialized) {
      const fresh: SmoothedFit = { ...target, initialized: true };
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

    return previous;
  }

  // -----------------------------------------------------------------------
  // drawing
  // -----------------------------------------------------------------------

  private drawHelmet(
    ctx: CanvasRenderingContext2D,
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
    const edge = Math.max(1, height * 0.009);
    const pulse = this.reducedMotion
      ? 1
      : 0.88 + 0.12 * Math.sin(time * 3.4);

    // --- helmet shell (red) ------------------------------------------------
    traceShell(ctx, halfWidth, ty);
    const shell = ctx.createLinearGradient(0, top, 0, bottom);
    shell.addColorStop(0, '#e2464c');
    shell.addColorStop(0.24, '#b81c25');
    shell.addColorStop(0.62, '#7c0c13');
    shell.addColorStop(1, '#390408');
    ctx.fillStyle = shell;
    ctx.fill();
    ctx.lineWidth = edge * 1.5;
    ctx.strokeStyle = 'rgba(18, 3, 5, 0.9)';
    ctx.stroke();

    // Dome sheen — clipped to the shell so it never spills outside.
    ctx.save();
    ctx.clip();
    const sheen = ctx.createRadialGradient(
      -halfWidth * 0.35,
      ty(0.1),
      0,
      -halfWidth * 0.35,
      ty(0.1),
      halfWidth * 1.5,
    );
    sheen.addColorStop(0, 'rgba(255, 220, 220, 0.32)');
    sheen.addColorStop(0.45, 'rgba(255, 255, 255, 0.04)');
    sheen.addColorStop(1, 'rgba(0, 0, 0, 0.3)');
    ctx.fillStyle = sheen;
    ctx.fillRect(-halfWidth * 1.1, top, halfWidth * 2.2, height);
    ctx.restore();

    // --- ear pods ----------------------------------------------------------
    for (const side of [-1, 1] as const) {
      const cx = side * halfWidth * 0.8;
      const cy = ty(0.46);
      const radius = height * 0.072;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, TAU);
      ctx.fillStyle = '#5c0a10';
      ctx.fill();
      ctx.lineWidth = edge * 1.3;
      ctx.strokeStyle = '#c99a3f';
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(cx, cy, radius * 0.44, 0, TAU);
      ctx.fillStyle = '#22060a';
      ctx.fill();
    }

    // --- gold faceplate ----------------------------------------------------
    traceFaceplate(ctx, halfWidth, ty);
    const gold = ctx.createLinearGradient(0, ty(0.08), 0, ty(1));
    gold.addColorStop(0, '#f9e6a4');
    gold.addColorStop(0.22, '#e7bd5f');
    gold.addColorStop(0.55, '#c2923a');
    gold.addColorStop(0.82, '#8d611c');
    gold.addColorStop(1, '#5e3c0e');
    ctx.fillStyle = gold;
    ctx.fill();
    ctx.lineWidth = edge * 1.6;
    ctx.strokeStyle = '#3f2606';
    ctx.stroke();

    // Faceplate shading + engraved panel lines, clipped to the plate.
    ctx.save();
    ctx.clip();

    const plateSheen = ctx.createLinearGradient(
      -halfWidth,
      ty(0.2),
      halfWidth,
      ty(0.9),
    );
    plateSheen.addColorStop(0, 'rgba(255, 255, 255, 0.26)');
    plateSheen.addColorStop(0.35, 'rgba(255, 255, 255, 0.04)');
    plateSheen.addColorStop(0.7, 'rgba(0, 0, 0, 0.1)');
    plateSheen.addColorStop(1, 'rgba(0, 0, 0, 0.28)');
    ctx.fillStyle = plateSheen;
    ctx.fillRect(-halfWidth, top, halfWidth * 2, height);

    ctx.strokeStyle = 'rgba(64, 38, 6, 0.7)';
    ctx.lineWidth = edge;
    ctx.lineCap = 'round';
    ctx.beginPath();

    // Brow seam.
    ctx.moveTo(-halfWidth * 0.52, ty(0.34));
    ctx.quadraticCurveTo(0, ty(0.24), halfWidth * 0.52, ty(0.34));
    // Centre seam down the forehead.
    ctx.moveTo(0, ty(0.1));
    ctx.lineTo(0, ty(0.37));

    // Cheek creases and nose-ridge lines.
    for (const side of [-1, 1] as const) {
      ctx.moveTo(side * halfWidth * 0.6, ty(0.4));
      ctx.quadraticCurveTo(
        side * halfWidth * 0.48,
        ty(0.62),
        side * halfWidth * 0.3,
        ty(0.84),
      );

      ctx.moveTo(side * halfWidth * 0.1, ty(0.42));
      ctx.quadraticCurveTo(
        side * halfWidth * 0.07,
        ty(0.5),
        side * halfWidth * 0.05,
        ty(0.57),
      );
    }

    // Nose base.
    ctx.moveTo(-halfWidth * 0.09, ty(0.585));
    ctx.quadraticCurveTo(0, ty(0.61), halfWidth * 0.09, ty(0.585));
    ctx.stroke();

    // Chin seam.
    ctx.beginPath();
    ctx.moveTo(-halfWidth * 0.2, ty(0.93));
    ctx.quadraticCurveTo(0, ty(0.975), halfWidth * 0.2, ty(0.93));
    ctx.stroke();
    ctx.restore();

    // --- glowing eye slits -------------------------------------------------
    this.drawEyeSlits(ctx, fit, halfWidth, height, edge, pulse);

    // --- mouth grill -------------------------------------------------------
    this.drawMouthGrill(ctx, fit, halfWidth, height, ty, edge);

    // --- nose tip glint ----------------------------------------------------
    const noseGlintY = clamp(fit.noseY, ty(0.42), ty(0.68));
    ctx.beginPath();
    ctx.ellipse(
      -halfWidth * 0.05,
      noseGlintY,
      halfWidth * 0.06,
      height * 0.018,
      -0.25,
      0,
      TAU,
    );
    ctx.fillStyle = 'rgba(255, 245, 200, 0.5)';
    ctx.fill();
  }

  private drawEyeSlits(
    ctx: CanvasRenderingContext2D,
    fit: IronManFit,
    halfWidth: number,
    height: number,
    edge: number,
    pulse: number,
  ): void {
    // Keep the slits inside the faceplate for very wide faces.
    const iod = Math.min(fit.eyeDistance, halfWidth * 1.1);
    const halfLength = iod * 0.19;
    const halfHeight = iod * 0.082;
    const y = 0;

    for (const side of [-1, 1] as const) {
      const inner = side * (iod * 0.5 - halfLength);
      const outer = side * (iod * 0.5 + halfLength);
      const middle = (inner + outer) / 2;

      ctx.beginPath();
      ctx.moveTo(inner, y + halfHeight * 0.25);
      ctx.quadraticCurveTo(middle, y - halfHeight * 1.05, outer, y - halfHeight * 0.15);
      ctx.quadraticCurveTo(middle, y + halfHeight, inner, y + halfHeight * 0.25);
      ctx.closePath();

      // Dark socket.
      ctx.fillStyle = '#0d1217';
      ctx.fill();

      // Cyan glow + bright core.
      ctx.save();
      ctx.shadowColor = `rgba(150, 240, 255, ${0.85 * pulse})`;
      ctx.shadowBlur = halfHeight * 2.8;
      ctx.fillStyle = `rgba(214, 249, 255, ${0.92 * pulse})`;
      ctx.fill();
      ctx.restore();

      ctx.lineWidth = Math.max(0.8, edge * 0.75);
      ctx.strokeStyle = 'rgba(180, 245, 255, 0.85)';
      ctx.stroke();
    }

    // Thin dark brow shadow above each slit for contrast against the gold.
    ctx.strokeStyle = 'rgba(60, 36, 6, 0.55)';
    ctx.lineWidth = Math.max(1, height * 0.008);
    ctx.beginPath();
    for (const side of [-1, 1] as const) {
      const inner = side * (iod * 0.5 - halfLength);
      const outer = side * (iod * 0.5 + halfLength);
      ctx.moveTo(inner, -halfHeight * 1.15);
      ctx.quadraticCurveTo(
        (inner + outer) / 2,
        -halfHeight * 2.2,
        outer,
        -halfHeight * 1.3,
      );
    }
    ctx.stroke();
  }

  private drawMouthGrill(
    ctx: CanvasRenderingContext2D,
    fit: IronManFit,
    halfWidth: number,
    height: number,
    ty: (t: number) => number,
    edge: number,
  ): void {
    const halfMouth = clamp(
      fit.mouthWidth * 0.6,
      halfWidth * 0.18,
      halfWidth * 0.36,
    );
    const grillHalfHeight = height * 0.052;
    const mouthY = clamp(fit.mouthY, ty(0.7), ty(0.9));

    roundRectPath(
      ctx,
      -halfMouth,
      mouthY - grillHalfHeight,
      halfMouth * 2,
      grillHalfHeight * 2,
      grillHalfHeight * 0.5,
    );
    ctx.fillStyle = '#22130a';
    ctx.fill();
    ctx.lineWidth = edge;
    ctx.strokeStyle = '#7a5119';
    ctx.stroke();

    // Ribs inside the grill.
    ctx.save();
    ctx.clip();
    ctx.strokeStyle = 'rgba(222, 181, 108, 0.5)';
    ctx.lineWidth = Math.max(1, grillHalfHeight * 0.16);
    ctx.beginPath();
    for (let i = 1; i <= 3; i++) {
      const ribY = mouthY - grillHalfHeight + (i / 4) * grillHalfHeight * 2;
      ctx.moveTo(-halfMouth * 0.86, ribY);
      ctx.lineTo(halfMouth * 0.86, ribY);
    }
    ctx.stroke();
    ctx.restore();
  }
}

// -------------------------------------------------------------------------
// path helpers
// -------------------------------------------------------------------------

function traceShell(
  ctx: CanvasRenderingContext2D,
  halfWidth: number,
  ty: (t: number) => number,
): void {
  ctx.beginPath();
  ctx.moveTo(0, ty(0));
  ctx.bezierCurveTo(
    halfWidth * 0.66,
    ty(-0.005),
    halfWidth * 1.02,
    ty(0.14),
    halfWidth,
    ty(0.4),
  );
  ctx.bezierCurveTo(
    halfWidth * 0.98,
    ty(0.55),
    halfWidth * 0.9,
    ty(0.67),
    halfWidth * 0.78,
    ty(0.78),
  );
  ctx.bezierCurveTo(
    halfWidth * 0.62,
    ty(0.94),
    halfWidth * 0.28,
    ty(1),
    0,
    ty(1),
  );
  ctx.bezierCurveTo(
    -halfWidth * 0.28,
    ty(1),
    -halfWidth * 0.62,
    ty(0.94),
    -halfWidth * 0.78,
    ty(0.78),
  );
  ctx.bezierCurveTo(
    -halfWidth * 0.9,
    ty(0.67),
    -halfWidth * 0.98,
    ty(0.55),
    -halfWidth,
    ty(0.4),
  );
  ctx.bezierCurveTo(
    -halfWidth * 1.02,
    ty(0.14),
    -halfWidth * 0.66,
    ty(-0.005),
    0,
    ty(0),
  );
  ctx.closePath();
}

function traceFaceplate(
  ctx: CanvasRenderingContext2D,
  halfWidth: number,
  ty: (t: number) => number,
): void {
  ctx.beginPath();
  ctx.moveTo(0, ty(0.085));
  ctx.bezierCurveTo(
    halfWidth * 0.34,
    ty(0.075),
    halfWidth * 0.58,
    ty(0.15),
    halfWidth * 0.68,
    ty(0.29),
  );
  ctx.bezierCurveTo(
    halfWidth * 0.77,
    ty(0.43),
    halfWidth * 0.74,
    ty(0.6),
    halfWidth * 0.60,
    ty(0.74),
  );
  ctx.bezierCurveTo(
    halfWidth * 0.46,
    ty(0.87),
    halfWidth * 0.32,
    ty(0.955),
    halfWidth * 0.18,
    ty(0.985),
  );
  ctx.lineTo(-halfWidth * 0.18, ty(0.985));
  ctx.bezierCurveTo(
    -halfWidth * 0.32,
    ty(0.955),
    -halfWidth * 0.46,
    ty(0.87),
    -halfWidth * 0.60,
    ty(0.74),
  );
  ctx.bezierCurveTo(
    -halfWidth * 0.74,
    ty(0.6),
    -halfWidth * 0.77,
    ty(0.43),
    -halfWidth * 0.68,
    ty(0.29),
  );
  ctx.bezierCurveTo(
    -halfWidth * 0.58,
    ty(0.15),
    -halfWidth * 0.34,
    ty(0.075),
    0,
    ty(0.085),
  );
  ctx.closePath();
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
