import { describe, expect, it } from 'vitest';
import type { Point2D, Point3D } from '../src/app/types';
import { estimateIronManFit } from '../src/modules/IronManMask';

/** Build a sparse face-mesh array containing only the given landmarks. */
function faceWith(points: Record<number, [number, number]>): Point3D[] {
  const face: Point3D[] = [];
  for (const [index, [x, y]] of Object.entries(points)) {
    face[Number(index)] = { x, y, z: 0 };
  }
  return face;
}

/** Unmirrored 1000×1000 conversion. */
const toCanvas = (p: Point3D): Point2D => ({ x: p.x * 1000, y: p.y * 1000 });

/** Mirrored 1000×1000 conversion (selfie view). */
const toCanvasMirrored = (p: Point3D): Point2D => ({
  x: (1 - p.x) * 1000,
  y: p.y * 1000,
});

/** Upright face: forehead 250, chin 750, eye line at 500, cheeks at ±200. */
const BASE_FACE: Record<number, [number, number]> = {
  10: [0.5, 0.25], // forehead top
  152: [0.5, 0.75], // chin
  234: [0.3, 0.5], // right cheek
  454: [0.7, 0.5], // left cheek
  33: [0.4, 0.5], // right outer eye
  263: [0.6, 0.5], // left outer eye
};

describe('estimateIronManFit', () => {
  it('derives the upright fit from the core landmarks', () => {
    const fit = estimateIronManFit(faceWith(BASE_FACE), toCanvas);
    expect(fit).not.toBeNull();
    if (!fit) return;

    expect(fit.center).toEqual({ x: 500, y: 500 });
    expect(fit.roll).toBe(0);
    expect(fit.faceWidth).toBe(400);
    expect(fit.faceHeight).toBe(500);
    expect(fit.eyeDistance).toBe(200);
    expect(fit.browY).toBe(-250);
    expect(fit.chinY).toBe(250);
  });

  it('uses mouth and nose landmarks when they are present', () => {
    const fit = estimateIronManFit(
      faceWith({
        ...BASE_FACE,
        13: [0.5, 0.65], // upper lip
        14: [0.5, 0.67], // lower lip
        61: [0.44, 0.66], // right mouth corner
        291: [0.56, 0.66], // left mouth corner
        1: [0.5, 0.6], // nose tip
      }),
      toCanvas,
    );
    expect(fit).not.toBeNull();
    if (!fit) return;

    expect(fit.mouthY).toBeCloseTo(160, 5);
    expect(fit.mouthWidth).toBeCloseTo(120, 5);
    expect(fit.noseY).toBeCloseTo(100, 5);
  });

  it('falls back to proportions when optional landmarks are missing', () => {
    const fit = estimateIronManFit(faceWith(BASE_FACE), toCanvas);
    expect(fit).not.toBeNull();
    if (!fit) return;

    // chinY = 250 → mouthY = 145, mouthWidth = 400 · 0.3 = 120, noseY = 87.5.
    expect(fit.mouthY).toBeCloseTo(145, 5);
    expect(fit.mouthWidth).toBeCloseTo(120, 5);
    expect(fit.noseY).toBeCloseTo(87.5, 5);
  });

  it('follows head roll and expresses anchors in the rotated frame', () => {
    const fit = estimateIronManFit(
      faceWith({
        ...BASE_FACE,
        33: [0.4, 0.45],
        263: [0.6, 0.55],
      }),
      toCanvas,
    );
    expect(fit).not.toBeNull();
    if (!fit) return;

    const expectedRoll = Math.atan2(100, 200);
    expect(fit.center).toEqual({ x: 500, y: 500 });
    expect(fit.roll).toBeCloseTo(expectedRoll, 5);
    // The forehead is straight up, so its rotated y is -250 · cos(roll).
    expect(fit.browY).toBeCloseTo(-250 * Math.cos(expectedRoll), 5);
    expect(fit.chinY).toBeCloseTo(250 * Math.cos(expectedRoll), 5);
  });

  it('flips the roll sign in the mirrored selfie view', () => {
    const tilted = {
      ...BASE_FACE,
      33: [0.4, 0.45],
      263: [0.6, 0.55],
    } as Record<number, [number, number]>;

    const unmirrored = estimateIronManFit(faceWith(tilted), toCanvas);
    const mirrored = estimateIronManFit(faceWith(tilted), toCanvasMirrored);
    expect(unmirrored).not.toBeNull();
    expect(mirrored).not.toBeNull();
    if (!unmirrored || !mirrored) return;

    expect(mirrored.center.x).toBeCloseTo(500, 5);
    expect(mirrored.roll).toBeCloseTo(-unmirrored.roll, 5);
  });

  it('estimates head yaw from the nose-tip offset', () => {
    const fit = estimateIronManFit(
      faceWith({ ...BASE_FACE, 1: [0.55, 0.6] }), // nose tip 50 px right
      toCanvas,
    );
    expect(fit).not.toBeNull();
    if (!fit) return;

    // 50 px offset against a 200 px half face width.
    expect(fit.yaw).toBeCloseTo(0.25, 5);
  });

  it('clamps extreme head yaw for lighting purposes', () => {
    const fit = estimateIronManFit(
      faceWith({ ...BASE_FACE, 1: [0.95, 0.6] }),
      toCanvas,
    );
    expect(fit).not.toBeNull();
    if (!fit) return;

    expect(fit.yaw).toBe(0.85);
  });

  it('is yaw-neutral without a usable nose landmark', () => {
    const fit = estimateIronManFit(faceWith(BASE_FACE), toCanvas);
    expect(fit).not.toBeNull();
    if (!fit) return;

    expect(fit.yaw).toBe(0);
  });

  it('flips the yaw sign in the mirrored selfie view', () => {
    const turned = {
      ...BASE_FACE,
      1: [0.55, 0.6],
    } as Record<number, [number, number]>;

    const unmirrored = estimateIronManFit(faceWith(turned), toCanvas);
    const mirrored = estimateIronManFit(faceWith(turned), toCanvasMirrored);
    expect(unmirrored).not.toBeNull();
    expect(mirrored).not.toBeNull();
    if (!unmirrored || !mirrored) return;

    expect(mirrored.yaw).toBeCloseTo(-unmirrored.yaw, 5);
  });

  it('returns null for missing, degenerate or tiny faces', () => {
    expect(estimateIronManFit([], toCanvas)).toBeNull();
    // Missing eye landmarks.
    expect(
      estimateIronManFit(
        faceWith({
          10: [0.5, 0.25],
          152: [0.5, 0.75],
          234: [0.3, 0.5],
          454: [0.7, 0.5],
        }),
        toCanvas,
      ),
    ).toBeNull();
    // Sub-pixel face.
    expect(
      estimateIronManFit(
        faceWith({
          10: [0.5, 0.499],
          152: [0.5, 0.501],
          234: [0.4995, 0.5],
          454: [0.5005, 0.5],
          33: [0.4998, 0.5],
          263: [0.5002, 0.5],
        }),
        toCanvas,
      ),
    ).toBeNull();
  });
});
