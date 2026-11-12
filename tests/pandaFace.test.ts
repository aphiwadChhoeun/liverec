import { describe, expect, it } from 'vitest';
import type { Point2D, Point3D } from '../src/app/types';
import { estimatePandaFit } from '../src/modules/PandaFace';

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

describe('estimatePandaFit', () => {
  it('derives the upright fit from the core landmarks', () => {
    const fit = estimatePandaFit(faceWith(BASE_FACE), toCanvas);
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

  it('falls back to the eye spacing without inner eye corners', () => {
    const fit = estimatePandaFit(faceWith(BASE_FACE), toCanvas);
    expect(fit).not.toBeNull();
    if (!fit) return;

    expect(fit.eyeRight).toEqual({ x: -100, y: 0 });
    expect(fit.eyeLeft).toEqual({ x: 100, y: 0 });
  });

  it('places the patches on the inner/outer eye corner midpoints', () => {
    const fit = estimatePandaFit(
      faceWith({
        ...BASE_FACE,
        133: [0.46, 0.49], // right inner eye
        362: [0.54, 0.49], // left inner eye
      }),
      toCanvas,
    );
    expect(fit).not.toBeNull();
    if (!fit) return;

    expect(fit.eyeRight.x).toBeCloseTo(-70, 5);
    expect(fit.eyeRight.y).toBeCloseTo(-5, 5);
    expect(fit.eyeLeft.x).toBeCloseTo(70, 5);
    expect(fit.eyeLeft.y).toBeCloseTo(-5, 5);
  });

  it('uses mouth and nose landmarks when they are present', () => {
    const fit = estimatePandaFit(
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
    const fit = estimatePandaFit(faceWith(BASE_FACE), toCanvas);
    expect(fit).not.toBeNull();
    if (!fit) return;

    // chinY = 250 → mouthY = 145, mouthWidth = 400 · 0.3 = 120, noseY = 87.5.
    expect(fit.mouthY).toBeCloseTo(145, 5);
    expect(fit.mouthWidth).toBeCloseTo(120, 5);
    expect(fit.noseY).toBeCloseTo(87.5, 5);
  });

  it('follows head roll and rotates the eye anchors', () => {
    const fit = estimatePandaFit(
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
    // Eyes lie on the folded eye line, one half-spacing either side.
    expect(fit.eyeDistance).toBeCloseTo(Math.hypot(200, 100), 5);
    expect(fit.eyeRight.x).toBeCloseTo(-fit.eyeDistance / 2, 5);
    expect(fit.eyeRight.y).toBeCloseTo(0, 5);
    expect(fit.eyeLeft.x).toBeCloseTo(fit.eyeDistance / 2, 5);
    expect(fit.eyeLeft.y).toBeCloseTo(0, 5);
  });

  it('flips the roll sign in the mirrored selfie view', () => {
    const tilted = {
      ...BASE_FACE,
      33: [0.4, 0.45],
      263: [0.6, 0.55],
    } as Record<number, [number, number]>;

    const unmirrored = estimatePandaFit(faceWith(tilted), toCanvas);
    const mirrored = estimatePandaFit(faceWith(tilted), toCanvasMirrored);
    expect(unmirrored).not.toBeNull();
    expect(mirrored).not.toBeNull();
    if (!unmirrored || !mirrored) return;

    expect(mirrored.center.x).toBeCloseTo(500, 5);
    expect(mirrored.roll).toBeCloseTo(-unmirrored.roll, 5);
  });

  it('returns null for missing, degenerate or tiny faces', () => {
    expect(estimatePandaFit([], toCanvas)).toBeNull();
    // Missing eye landmarks.
    expect(
      estimatePandaFit(
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
      estimatePandaFit(
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
