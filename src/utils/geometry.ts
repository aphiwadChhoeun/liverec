/**
 * Pure landmark-geometry helpers shared by the renderer, augmentation modules
 * and the unit tests. No DOM types are required except the duck-typed
 * `PathLike`, which keeps `traceFaceOval` testable outside a browser.
 */
import type { Point2D, Point3D } from '../app/types';

/** Axis-aligned box in canvas pixel space. */
export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Ordered loop of MediaPipe canonical face-mesh indices tracing the face oval
 * (forehead → right side → chin → left side → back to forehead). Derived from
 * MediaPipe's `FACEMESH_FACE_OVAL` connection set; indices are stable across
 * model versions.
 */
export const FACE_OVAL_LOOP: readonly number[] = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
  378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
  162, 21, 54, 103, 67, 109,
];

/** BlazePose wrist landmark indices (subject's left / right wrist). */
export const POSE_WRIST_INDICES: readonly number[] = [15, 16];

/** Wrist is landmark 0 of every hand. */
export const HAND_WRIST_INDEX = 0;

/** Fingertip landmark indices (thumb → pinky). */
export const HAND_FINGERTIP_INDICES: readonly number[] = [4, 8, 12, 16, 20];

/** Convert a normalized landmark to canvas pixels, honoring mirroring. */
export function toPixel(
  point: Point3D,
  width: number,
  height: number,
  mirrored: boolean,
): Point2D {
  return {
    x: (mirrored ? 1 - point.x : point.x) * width,
    y: point.y * height,
  };
}

/**
 * Visibility gate for landmarks. Landmarks without a visibility score (e.g.
 * face-mesh points) are treated as fully visible; pass a threshold of `0` to
 * accept everything.
 */
export function pointVisible(point: Point3D, minVisibility = 0.5): boolean {
  return (point.visibility ?? 1) >= minVisibility;
}

/** Bounding box of a landmark set in canvas pixels, or `null` when empty. */
export function boundingBox(
  points: readonly Point3D[],
  width: number,
  height: number,
  mirrored: boolean,
): BoundingBox | null {
  if (points.length === 0) return null;
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const p of points) {
    const x = (mirrored ? 1 - p.x : p.x) * width;
    const y = p.y * height;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
  }
  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
}

/** Mean position of a 2D point set, or `null` when empty. */
export function centroid2D(points: readonly Point2D[]): Point2D | null {
  if (points.length === 0) return null;
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += p.x;
    sy += p.y;
  }
  return { x: sx / points.length, y: sy / points.length };
}

export function distance2D(a: Point2D, b: Point2D): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}

export function distance3D(a: Point3D, b: Point3D): number {
  return Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Linear interpolation between two landmarks (weights blend the z axis too). */
export function lerpPoint3D(a: Point3D, b: Point3D, t: number): Point3D {
  const blended: Point3D = {
    x: lerp(a.x, b.x, t),
    y: lerp(a.y, b.y, t),
    z: lerp(a.z, b.z, t),
  };
  const va = a.visibility;
  const vb = b.visibility;
  if (va !== undefined || vb !== undefined) {
    blended.visibility = lerp(va ?? 1, vb ?? 1, t);
  }
  return blended;
}

/**
 * Minimal path contract (satisfied by `Path2D`) so this helper runs in
 * Node-based unit tests without a DOM.
 */
export interface PathLike {
  moveTo(x: number, y: number): void;
  lineTo(x: number, y: number): void;
  closePath(): void;
}

export interface TraceOvalOptions {
  /** Mirror the X axis (selfie view). Default `false`. */
  mirrored?: boolean;
  /** Scale the oval around its centroid (`1` = exact fit, `>1` = padding). Default `1`. */
  scale?: number;
}

/**
 * Trace the face oval of one face onto `path` in canvas pixel coordinates.
 * Returns `false` (path untouched) when the face has too few oval landmarks.
 */
export function traceFaceOval(
  path: PathLike,
  faceLandmarks: readonly Point3D[],
  width: number,
  height: number,
  options: TraceOvalOptions = {},
): boolean {
  const mirrored = options.mirrored ?? false;
  const scale = options.scale ?? 1;

  const oval: Point2D[] = [];
  for (const index of FACE_OVAL_LOOP) {
    const lm = faceLandmarks[index];
    if (!lm) continue;
    oval.push(toPixel(lm, width, height, mirrored));
  }
  if (oval.length < 3) return false;

  const center = centroid2D(oval);
  if (!center) return false;

  const scaled = (p: Point2D): Point2D => ({
    x: p.x + (p.x - center.x) * (scale - 1),
    y: p.y + (p.y - center.y) * (scale - 1),
  });

  const first = scaled(oval[0]!);
  path.moveTo(first.x, first.y);
  for (let i = 1; i < oval.length; i++) {
    const p = scaled(oval[i]!);
    path.lineTo(p.x, p.y);
  }
  path.closePath();
  return true;
}
