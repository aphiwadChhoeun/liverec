/**
 * Panda face augmentation.
 *
 * Fits a cartoon panda onto the tracked face using a handful of stable
 * MediaPipe face-mesh landmarks (forehead, chin, cheekbones, eye corners, nose
 * and mouth) and draws the panda features in a local, roll-aligned space
 * (translate to the midpoint of the eyes, rotate by the eye-line angle):
 *
 *   1. ears        — soft black discs peeking above the head, drawn first so
 *                    they sit "behind" the face
 *   2. muzzle      — a warm-white wash around the nose/mouth that reads as fur
 *   3. eye patches — slanted black patches built with an even-odd hole around
 *                    each eye, so the wearer's eyes stay visible
 *   4. nose        — glossy black button nose over the nose tip
 *   5. mouth       — the little panda "w" smile below the nose
 *
 * The fit is EMA-smoothed across frames so the patches do not jitter with the
 * raw landmarks, and the selfie mirroring is honored through `context.toCanvas`.
 * A short appear ramp fades the panda in when a face is acquired; the same
 * ramp runs backwards when the face leaves the frame, masking single-frame
 * detection drop-outs. Transforms and shadows are always restored so later
 * modules inherit a clean context.
 *
 * Performance: `Path2D` shapes and gradients are cached per face and rebuilt
 * only when the quantized fit changes, so a steady face costs almost no path
 * building per frame.
 */
import type { AugmentationModule, ModuleContext } from './AugmentationModule';
import type { DetectionResult, Point2D, Point3D } from '../app/types';

/** Face-mesh indices the panda is anchored to. */
export const PANDA_FACE_INDICES = {
  foreheadTop: 10,
  chin: 152,
  cheekRight: 234,
  cheekLeft: 454,
  eyeOuterRight: 33,
  eyeInnerRight: 133,
  eyeOuterLeft: 263,
  eyeInnerLeft: 362,
  noseTip: 1,
  mouthUpper: 13,
  mouthLower: 14,
  mouthCornerRight: 61,
  mouthCornerLeft: 291,
} as const;

/** Below this many canvas pixels a face is too small to fit a legible panda. */
const MIN_FACE_PX = 24;

/** Appear / fade timings (seconds). */
const APPEAR_T = 0.26;
const APPEAR_T_REDUCED = 0.12;
const FADE_T = 0.22;

/** Even-odd hole around each eye, as a fraction of the patch size. */
const EYE_HOLE_RX = 0.68;
const EYE_HOLE_RY = 0.62;

/** Outward slant of the eye patches (radians). */
const PATCH_SLANT = 0.34;

const TAU = Math.PI * 2;

/** Immutable per-frame fit of the panda, in canvas pixel space. */
export interface PandaFit {
  /** Midpoint of the outer eye corners. */
  center: Point2D;
  /** Eye-line angle in radians, normalized to [-π/2, π/2]. */
  roll: number;
  /** Cheekbone-to-cheekbone width. */
  faceWidth: number;
  /** Forehead-to-chin height. */
  faceHeight: number;
  /** Outer-eye-corner distance. */
  eyeDistance: number;
  /** Local y of the forehead landmark (negative = above). */
  browY: number;
  /** Local y of the chin landmark (positive = below). */
  chinY: number;
  /** Local y of the nose tip. */
  noseY: number;
  /** Local y of the mouth center. */
  mouthY: number;
  /** Corner-to-corner mouth width. */
  mouthWidth: number;
  /** Local position of the image-left eye (subject's right eye). */
  eyeRight: Point2D;
  /** Local position of the image-right eye (subject's left eye). */
  eyeLeft: Point2D;
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
 * Estimate the panda fit for one face. Pure apart from the supplied
 * `toCanvas` converter, which keeps it testable without a DOM.
 *
 * Returns `null` when the required landmarks are missing, non-finite or the
 * face is too small to draw legible panda features.
 */
export function estimatePandaFit(
  face: readonly Point3D[],
  toCanvas: (point: Point3D) => Point2D,
): PandaFit | null {
  const forehead = face[PANDA_FACE_INDICES.foreheadTop];
  const chin = face[PANDA_FACE_INDICES.chin];
  const cheekRight = face[PANDA_FACE_INDICES.cheekRight];
  const cheekLeft = face[PANDA_FACE_INDICES.cheekLeft];
  const eyeOuterRight = face[PANDA_FACE_INDICES.eyeOuterRight];
  const eyeOuterLeft = face[PANDA_FACE_INDICES.eyeOuterLeft];
  if (
    !forehead ||
    !chin ||
    !cheekRight ||
    !cheekLeft ||
    !eyeOuterRight ||
    !eyeOuterLeft
  ) {
    return null;
  }

  const pForehead = toCanvas(forehead);
  const pChin = toCanvas(chin);
  const pCheekRight = toCanvas(cheekRight);
  const pCheekLeft = toCanvas(cheekLeft);
  const pEyeOuterRight = toCanvas(eyeOuterRight);
  const pEyeOuterLeft = toCanvas(eyeOuterLeft);

  const center: Point2D = {
    x: (pEyeOuterRight.x + pEyeOuterLeft.x) / 2,
    y: (pEyeOuterRight.y + pEyeOuterLeft.y) / 2,
  };

  // Eye-line angle, folded into [-π/2, π/2]: the panda is left/right
  // symmetric, so a 180° flip would wrongly turn it upside down.
  let roll = Math.atan2(
    pEyeOuterLeft.y - pEyeOuterRight.y,
    pEyeOuterLeft.x - pEyeOuterRight.x,
  );
  while (roll > Math.PI / 2) roll -= Math.PI;
  while (roll < -Math.PI / 2) roll += Math.PI;

  const faceWidth = distance(pCheekRight, pCheekLeft);
  const faceHeight = distance(pForehead, pChin);
  const eyeDistance = distance(pEyeOuterRight, pEyeOuterLeft);
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

  // Rotate landmarks into the roll-aligned local frame (y grows downward).
  const cos = Math.cos(roll);
  const sin = Math.sin(roll);
  const local = (p: Point2D): Point2D => {
    const dx = p.x - center.x;
    const dy = p.y - center.y;
    return {
      x: dx * cos + dy * sin,
      y: -dx * sin + dy * cos,
    };
  };

  const browY = local(pForehead).y;
  const chinY = local(pChin).y;

  // Eye centers come from the inner/outer corners when available, so the
  // patches track the actual eyes rather than the idealized face width.
  const eyeCenter = (
    outerIndex: number,
    innerIndex: number,
    fallbackX: number,
  ): Point2D => {
    const outer = face[outerIndex];
    const inner = face[innerIndex];
    if (outer && inner) {
      const a = local(toCanvas(outer));
      const b = local(toCanvas(inner));
      const point = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
      if (Number.isFinite(point.x) && Number.isFinite(point.y)) return point;
    }
    return { x: fallbackX, y: 0 };
  };

  const eyeRight = eyeCenter(
    PANDA_FACE_INDICES.eyeOuterRight,
    PANDA_FACE_INDICES.eyeInnerRight,
    -eyeDistance / 2,
  );
  const eyeLeft = eyeCenter(
    PANDA_FACE_INDICES.eyeOuterLeft,
    PANDA_FACE_INDICES.eyeInnerLeft,
    eyeDistance / 2,
  );

  const mouthUpper = face[PANDA_FACE_INDICES.mouthUpper];
  const mouthLower = face[PANDA_FACE_INDICES.mouthLower];
  const mouthY = finiteOr(
    mouthUpper && mouthLower
      ? (local(toCanvas(mouthUpper)).y + local(toCanvas(mouthLower)).y) / 2
      : chinY * 0.58,
    chinY * 0.58,
  );

  const mouthCornerRight = face[PANDA_FACE_INDICES.mouthCornerRight];
  const mouthCornerLeft = face[PANDA_FACE_INDICES.mouthCornerLeft];
  const mouthWidth = finiteOr(
    mouthCornerRight && mouthCornerLeft
      ? distance(toCanvas(mouthCornerRight), toCanvas(mouthCornerLeft))
      : faceWidth * 0.3,
    faceWidth * 0.3,
  );

  const noseTip = face[PANDA_FACE_INDICES.noseTip];
  const noseY = finiteOr(
    noseTip ? local(toCanvas(noseTip)).y : chinY * 0.35,
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
    noseY,
    mouthY,
    mouthWidth,
    eyeRight,
    eyeLeft,
  };
}

/** Derived, per-frame geometry shared by every paint layer. */
interface PandaMetrics {
  faceWidth: number;
  faceHeight: number;
  /** Base stroke weight, scaled with the face. */
  edge: number;
  /** Eye-patch center and extent, in local space. */
  patches: ReadonlyArray<{
    center: Point2D;
    /** Center of the see-through eye hole (the actual eye). */
    holeCenter: Point2D;
    rx: number;
    ry: number;
    angle: number;
  }>;
  /** Ear geometry (one side is mirrored). */
  earX: number;
  earY: number;
  earR: number;
  /** Nose geometry. */
  noseY: number;
  noseW: number;
  noseH: number;
  /** Mouth geometry. */
  mouthY: number;
  mouthHalf: number;
  /** Muzzle wash center / radii. */
  muzzleY: number;
  muzzleRX: number;
  muzzleRY: number;
}

/**
 * Single-slot paint cache per face: paths and gradients are rebuilt only when
 * the quantized geometry key changes. `CanvasGradient` and `Path2D` are not
 * bound to a context, so they are safe to reuse across frames.
 */
class PandaPaint {
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

interface PandaState extends PandaFit {
  /** 0→1 appear ramp; decays back to 0 when the face is lost. */
  alpha: number;
  /** Geometry-keyed paint cache for this face. */
  paint: PandaPaint;
}

/** Exponential smoothing time constants (seconds). Lower = snappier. */
const SMOOTH_TAU = {
  center: 0.05,
  roll: 0.07,
  size: 0.09,
  feature: 0.07,
  eye: 0.06,
} as const;

export class PandaFace implements AugmentationModule {
  id = 'panda-face';
  name = 'Panda face';
  description = 'Panda ears, eye patches and nose tracked onto the face.';
  enabled = false;

  private readonly states: Array<PandaState | undefined> = [];
  private readonly reducedMotion: boolean;

  constructor(options: { enabled?: boolean } = {}) {
    this.enabled = options.enabled ?? false;
    let reduced = false;
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }
    this.reducedMotion = reduced;
    if (reduced) {
      this.description = 'Panda face — pop-in animation disabled (reduced motion).';
    }
  }

  update(
    ctx: CanvasRenderingContext2D,
    result: DetectionResult,
    context: ModuleContext,
  ): void {
    const dt = clamp(context.dt, 0, 0.1); // tab switches produce huge gaps
    const faces = result.face?.landmarks ?? [];
    const appearRate = dt / (this.reducedMotion ? APPEAR_T_REDUCED : APPEAR_T);

    const count = Math.max(faces.length, this.states.length);
    for (let i = 0; i < count; i++) {
      const face = i < faces.length ? faces[i] : undefined;
      const target = face ? estimatePandaFit(face, context.toCanvas) : null;
      let state = this.states[i];

      if (target) {
        if (!state) {
          state = { ...target, alpha: 0, paint: new PandaPaint() };
          this.states[i] = state;
        } else {
          this.smooth(state, target, dt);
        }
        state.alpha = Math.min(1, state.alpha + appearRate);
      } else if (state) {
        // Face lost (or momentarily unfittable): fade out instead of blinking.
        state.alpha -= dt / FADE_T;
        if (state.alpha <= 0) {
          this.states[i] = undefined;
          continue;
        }
      } else {
        continue;
      }

      const ease = easeOutCubic(state.alpha);
      const scale = this.reducedMotion ? 1 : 0.95 + 0.05 * ease;

      ctx.save();
      try {
        ctx.translate(state.center.x, state.center.y);
        ctx.rotate(state.roll);
        ctx.scale(scale, scale);
        ctx.globalAlpha *= ease;
        this.paintPanda(ctx, state.paint, state);
      } finally {
        ctx.restore(); // never leak transform / shadows to the next module
      }
    }

    // Drop trailing states so a shrinking face count does not leak slots.
    while (
      this.states.length > faces.length &&
      !this.states[this.states.length - 1]
    ) {
      this.states.pop();
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

  /** EMA-smooth a per-face fit so the patches do not shimmer frame to frame. */
  private smooth(state: PandaState, target: PandaFit, dt: number): void {
    const alpha = (tau: number): number =>
      dt <= 0 ? 1 : 1 - Math.exp(-dt / tau);

    const aCenter = alpha(SMOOTH_TAU.center);
    state.center.x += (target.center.x - state.center.x) * aCenter;
    state.center.y += (target.center.y - state.center.y) * aCenter;

    // Fold the roll delta into [-π/2, π/2] so ±90° tilts do not spin around.
    let deltaRoll = target.roll - state.roll;
    while (deltaRoll > Math.PI / 2) deltaRoll -= Math.PI;
    while (deltaRoll < -Math.PI / 2) deltaRoll += Math.PI;
    state.roll += deltaRoll * alpha(SMOOTH_TAU.roll);

    const aSize = alpha(SMOOTH_TAU.size);
    state.faceWidth += (target.faceWidth - state.faceWidth) * aSize;
    state.faceHeight += (target.faceHeight - state.faceHeight) * aSize;
    state.eyeDistance += (target.eyeDistance - state.eyeDistance) * aSize;

    const aFeature = alpha(SMOOTH_TAU.feature);
    state.browY += (target.browY - state.browY) * aFeature;
    state.chinY += (target.chinY - state.chinY) * aFeature;
    state.noseY += (target.noseY - state.noseY) * aFeature;
    state.mouthY += (target.mouthY - state.mouthY) * aFeature;
    state.mouthWidth += (target.mouthWidth - state.mouthWidth) * aFeature;

    const aEye = alpha(SMOOTH_TAU.eye);
    state.eyeRight.x += (target.eyeRight.x - state.eyeRight.x) * aEye;
    state.eyeRight.y += (target.eyeRight.y - state.eyeRight.y) * aEye;
    state.eyeLeft.x += (target.eyeLeft.x - state.eyeLeft.x) * aEye;
    state.eyeLeft.y += (target.eyeLeft.y - state.eyeLeft.y) * aEye;
  }

  private reset(): void {
    this.states.length = 0;
  }

  // -----------------------------------------------------------------------
  // painting
  // -----------------------------------------------------------------------

  private paintPanda(
    ctx: CanvasRenderingContext2D,
    paint: PandaPaint,
    fit: PandaFit,
  ): void {
    const m = measure(fit);
    if (!m) return;

    // Rebuild cached paths / gradients only when the quantized fit moved.
    paint.sync(
      `${Math.round(m.faceWidth)}:${Math.round(m.faceHeight)}:` +
        `${Math.round(m.earX)}:${Math.round(m.earY)}:${Math.round(m.earR)}:` +
        `${Math.round(m.noseY)}:${Math.round(m.noseW)}:${Math.round(m.noseH)}:` +
        `${Math.round(m.mouthY)}:${Math.round(m.mouthHalf)}:` +
        m.patches
          .map(
            (p) =>
              `${Math.round(p.center.x)},${Math.round(p.center.y)},` +
              `${Math.round(p.holeCenter.x)},${Math.round(p.holeCenter.y)},` +
              `${Math.round(p.rx)},${Math.round(p.ry)}`,
          )
          .join(':'),
    );

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    this.drawMuzzle(ctx, paint, m);
    this.drawEars(ctx, paint, m);
    this.drawEyePatches(ctx, paint, m);
    this.drawNose(ctx, paint, m);
    this.drawMouth(ctx, m);
  }

  /** Soft white fur wash around the nose and mouth. */
  private drawMuzzle(
    ctx: CanvasRenderingContext2D,
    paint: PandaPaint,
    m: PandaMetrics,
  ): void {
    ctx.fillStyle = paint.gradient('muzzle', ctx, (c) =>
      radialGradient(c, 0, m.muzzleY, m.muzzleRX, [
        [0, 'rgba(255, 249, 240, 0.22)'],
        [0.55, 'rgba(255, 249, 240, 0.1)'],
        [1, 'rgba(255, 249, 240, 0)'],
      ]),
    );
    ctx.beginPath();
    ctx.ellipse(0, m.muzzleY, m.muzzleRX, m.muzzleRY, 0, 0, TAU);
    ctx.fill();
  }

  private drawEars(
    ctx: CanvasRenderingContext2D,
    paint: PandaPaint,
    m: PandaMetrics,
  ): void {
    const earGradient = paint.gradient('ear', ctx, (c) =>
      radialGradient(c, -m.earR * 0.3, -m.earR * 0.35, m.earR * 1.25, [
        [0, '#3b3736'],
        [0.42, '#1d1919'],
        [1, '#0b0a0a'],
      ]),
    );

    for (const side of [-1, 1] as const) {
      ctx.save();
      ctx.translate(side * m.earX, m.earY);
      ctx.rotate(side * 0.1);

      // Cast shadow seats the ear behind the head silhouette.
      ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
      ctx.shadowBlur = m.earR * 0.3;
      ctx.shadowOffsetY = m.earR * 0.06;
      ctx.fillStyle = earGradient;
      ctx.beginPath();
      ctx.ellipse(0, 0, m.earR, m.earR * 1.06, 0, 0, TAU);
      ctx.fill();
      ctx.restore();

      // Recessed inner ear.
      ctx.save();
      ctx.translate(side * m.earX, m.earY);
      ctx.beginPath();
      ctx.ellipse(
        0,
        m.earR * 0.1,
        m.earR * 0.54,
        m.earR * 0.6,
        side * 0.1,
        0,
        TAU,
      );
      ctx.fillStyle = '#090808';
      ctx.fill();
      ctx.restore();
    }
  }

  private drawEyePatches(
    ctx: CanvasRenderingContext2D,
    paint: PandaPaint,
    m: PandaMetrics,
  ): void {
    const patchFill = '#151110';

    m.patches.forEach((patch, index) => {
      const path = paint.path(`patch${index}`, () =>
        buildPatchPath(
          patch.center,
          patch.holeCenter,
          patch.rx,
          patch.ry,
          patch.angle,
        ),
      );

      ctx.save();
      ctx.shadowColor = 'rgba(0, 0, 0, 0.28)';
      ctx.shadowBlur = Math.max(1, m.edge * 1.6);
      ctx.fillStyle = patchFill;
      // Even-odd punches the eye hole so the wearer's eyes stay visible.
      ctx.fill(path, 'evenodd');
      ctx.restore();

      // Soft fur edge (also traces the eye hole, reading as eyeliner).
      ctx.lineWidth = Math.max(1, m.edge * 0.9);
      ctx.strokeStyle = 'rgba(96, 88, 82, 0.24)';
      ctx.stroke(path);
    });
  }

  private drawNose(
    ctx: CanvasRenderingContext2D,
    paint: PandaPaint,
    m: PandaMetrics,
  ): void {
    const nose = paint.path('nose', () =>
      buildNosePath(m.noseY, m.noseW, m.noseH),
    );

    ctx.fillStyle = paint.gradient('noseFill', ctx, (c) =>
      linearGradient(
        c,
        0,
        m.noseY - m.noseH * 0.6,
        0,
        m.noseY + m.noseH * 0.6,
        [
          [0, '#332d2b'],
          [0.45, '#171313'],
          [1, '#080707'],
        ],
      ),
    );
    ctx.fill(nose);
    ctx.lineWidth = Math.max(1, m.edge * 0.8);
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.6)';
    ctx.stroke(nose);

    // Glossy button-nose highlight.
    ctx.beginPath();
    ctx.ellipse(
      -m.noseW * 0.16,
      m.noseY - m.noseH * 0.24,
      m.noseW * 0.16,
      m.noseH * 0.11,
      -0.4,
      0,
      TAU,
    );
    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.fill();
  }

  private drawMouth(
    ctx: CanvasRenderingContext2D,
    m: PandaMetrics,
  ): void {
    ctx.beginPath();
    // Philtrum from the nose down to the "w" smile.
    ctx.moveTo(0, m.noseY + m.noseH * 0.45);
    ctx.lineTo(0, m.mouthY);
    // Left arc.
    ctx.quadraticCurveTo(
      -m.mouthHalf * 0.16,
      m.mouthY + m.mouthHalf * 0.4,
      -m.mouthHalf,
      m.mouthY + m.mouthHalf * 0.06,
    );
    // Right arc.
    ctx.moveTo(0, m.mouthY);
    ctx.quadraticCurveTo(
      m.mouthHalf * 0.16,
      m.mouthY + m.mouthHalf * 0.4,
      m.mouthHalf,
      m.mouthY + m.mouthHalf * 0.06,
    );

    ctx.lineWidth = Math.max(1, m.edge * 1.2);
    ctx.strokeStyle = 'rgba(26, 20, 19, 0.9)';
    ctx.stroke();
  }
}

// -------------------------------------------------------------------------
// geometry
// -------------------------------------------------------------------------

/**
 * Turn a smoothed fit into concrete local-space geometry, or `null` when the
 * fit is degenerate / inverted and should not be drawn.
 */
function measure(fit: PandaFit): PandaMetrics | null {
  const { faceWidth, faceHeight } = fit;
  if (
    !Number.isFinite(faceWidth) ||
    !Number.isFinite(faceHeight) ||
    faceWidth < MIN_FACE_PX ||
    faceHeight < MIN_FACE_PX
  ) {
    return null;
  }

  // Size the patches from the actual eye-center spacing (which is narrower
  // than the outer-corner distance), so they never meet at the nose bridge.
  const eyeSpan = Math.max(
    distance(fit.eyeRight, fit.eyeLeft),
    MIN_FACE_PX / 3,
  );
  const patchRx = eyeSpan * 0.42;
  const patchRy = patchRx * 0.74;

  const patches = [
    { eye: fit.eyeRight, side: -1 as const },
    { eye: fit.eyeLeft, side: 1 as const },
  ].map(({ eye, side }) => ({
    center: {
      x: eye.x + side * eyeSpan * 0.08,
      y: eye.y - eyeSpan * 0.01,
    },
    holeCenter: eye,
    rx: patchRx,
    ry: patchRy,
    angle: -side * PATCH_SLANT,
  }));

  const noseW = faceWidth * 0.175;
  const noseH = noseW * 0.75;
  const noseY = clamp(fit.noseY, fit.browY + faceHeight * 0.3, fit.chinY - faceHeight * 0.22);

  const mouthHalf = clamp(
    fit.mouthWidth * 0.42,
    faceWidth * 0.08,
    faceWidth * 0.22,
  );
  const mouthY = clamp(
    fit.mouthY,
    noseY + noseH * 0.8,
    fit.chinY - faceHeight * 0.05,
  );

  const muzzleY = (noseY + fit.chinY) / 2;

  return {
    faceWidth,
    faceHeight,
    edge: Math.max(1, faceHeight * 0.008),
    patches,
    earX: faceWidth * 0.56,
    earY: fit.browY - faceHeight * 0.04,
    earR: Math.max(faceWidth * 0.22, faceHeight * 0.11),
    noseY,
    noseW,
    noseH,
    mouthY,
    mouthHalf,
    muzzleY,
    muzzleRX: faceWidth * 0.42,
    muzzleRY: faceHeight * 0.18,
  };
}

/** Slanted black eye patch with an even-odd hole around the eye. */
function buildPatchPath(
  center: Point2D,
  holeCenter: Point2D,
  rx: number,
  ry: number,
  angle: number,
): Path2D {
  const p = new Path2D();
  p.ellipse(center.x, center.y, rx, ry, angle, 0, TAU);
  p.ellipse(
    holeCenter.x,
    holeCenter.y,
    rx * EYE_HOLE_RX,
    ry * EYE_HOLE_RY,
    angle,
    0,
    TAU,
  );
  return p;
}

/** Rounded button nose, widest at the top and tapering to a soft point. */
function buildNosePath(y: number, width: number, height: number): Path2D {
  const halfW = width / 2;
  const top = y - height * 0.5;
  const bottom = y + height * 0.55;
  const p = new Path2D();
  p.moveTo(-halfW * 0.62, top + height * 0.1);
  p.quadraticCurveTo(0, top - height * 0.14, halfW * 0.62, top + height * 0.1);
  p.quadraticCurveTo(halfW * 0.96, top + height * 0.5, 0, bottom);
  p.quadraticCurveTo(-halfW * 0.96, top + height * 0.5, -halfW * 0.62, top + height * 0.1);
  p.closePath();
  return p;
}

// -------------------------------------------------------------------------
// gradient helpers
// -------------------------------------------------------------------------

type GradientStops = ReadonlyArray<readonly [number, string]>;

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
