import { describe, expect, it } from 'vitest';
import type { Point3D } from '../src/app/types';
import {
  FACE_OVAL_LOOP,
  HAND_FINGERTIP_INDICES,
  HAND_WRIST_INDEX,
  POSE_WRIST_INDICES,
  boundingBox,
  centroid2D,
  distance2D,
  distance3D,
  lerp,
  lerpPoint3D,
  pointVisible,
  toPixel,
  traceFaceOval,
  type PathLike,
} from '../src/utils/geometry';

/** Records path commands so assertions can be made without a DOM. */
class PathRecorder implements PathLike {
  readonly ops: string[] = [];

  moveTo(x: number, y: number): void {
    this.ops.push(`M ${x} ${y}`);
  }

  lineTo(x: number, y: number): void {
    this.ops.push(`L ${x} ${y}`);
  }

  closePath(): void {
    this.ops.push('Z');
  }
}

/**
 * 478-point face where oval position `p` sits at x = (p+1)/64. Denominators
 * are powers of two, so every coordinate, centroid and scaled value is exact
 * in IEEE-754 — no float fuzz in the assertions below.
 */
function makeSymmetricFace(): Point3D[] {
  const face: Point3D[] = Array.from({ length: 478 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0,
  }));
  FACE_OVAL_LOOP.forEach((index, position) => {
    face[index] = { x: (position + 1) / 64, y: 0.5, z: 0 };
  });
  return face;
}

describe('toPixel', () => {
  it('converts normalized coordinates to canvas pixels', () => {
    const point: Point3D = { x: 0.25, y: 0.5, z: 0 };
    expect(toPixel(point, 1000, 500, false)).toEqual({ x: 250, y: 250 });
  });

  it('flips x when mirrored', () => {
    const point: Point3D = { x: 0.25, y: 0.5, z: 0 };
    expect(toPixel(point, 200, 100, true)).toEqual({ x: 150, y: 50 });
  });

  it('mirroring maps a point to where its mirror-image lands unmirrored', () => {
    const point: Point3D = { x: 0.75, y: 0.5, z: 0 };
    const mirrorImage: Point3D = { x: 0.25, y: 0.5, z: 0 };
    expect(toPixel(point, 200, 100, true).x).toBe(
      toPixel(mirrorImage, 200, 100, false).x,
    );
  });
});

describe('pointVisible', () => {
  it('treats missing visibility as fully visible', () => {
    expect(pointVisible({ x: 0, y: 0, z: 0 })).toBe(true);
  });

  it('gates on the threshold', () => {
    const low: Point3D = { x: 0, y: 0, z: 0, visibility: 0.3 };
    const high: Point3D = { x: 0, y: 0, z: 0, visibility: 0.7 };
    expect(pointVisible(low, 0.5)).toBe(false);
    expect(pointVisible(high, 0.5)).toBe(true);
  });

  it('accepts everything at threshold 0 (face landmarks carry visibility 0)', () => {
    const faceLike: Point3D = { x: 0, y: 0, z: 0, visibility: 0 };
    expect(pointVisible(faceLike, 0)).toBe(true);
  });
});

describe('boundingBox', () => {
  const points: Point3D[] = [
    { x: 0.2, y: 0.1, z: 0 },
    { x: 0.6, y: 0.3, z: 0 },
    { x: 0.4, y: 0.8, z: 0 },
  ];

  it('computes the pixel-space box', () => {
    expect(boundingBox(points, 100, 100, false)).toEqual({
      x: 20,
      y: 10,
      width: 40,
      height: 70,
    });
  });

  it('mirrors the box', () => {
    // Mirrored xs: 80, 40, 60 → box spans 40..80.
    expect(boundingBox(points, 100, 100, true)).toEqual({
      x: 40,
      y: 10,
      width: 40,
      height: 70,
    });
  });

  it('returns null for empty input', () => {
    expect(boundingBox([], 100, 100, false)).toBeNull();
  });
});

describe('centroid2D / distances / lerp', () => {
  it('computes the centroid', () => {
    expect(centroid2D([{ x: 0, y: 0 }, { x: 10, y: 20 }])).toEqual({ x: 5, y: 10 });
    expect(centroid2D([])).toBeNull();
  });

  it('computes distances', () => {
    expect(distance2D({ x: 0, y: 0 }, { x: 3, y: 4 })).toBe(5);
    expect(distance3D({ x: 0, y: 0, z: 0 }, { x: 2, y: 3, z: 6 })).toBe(7);
  });

  it('lerps numbers and points (visibility blended when present)', () => {
    expect(lerp(0, 10, 0.25)).toBe(2.5);
    const blended = lerpPoint3D(
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 2, z: 3, visibility: 1 },
      0.5,
    );
    expect(blended).toEqual({ x: 0.5, y: 1, z: 1.5, visibility: 1 });
    const noVisibility = lerpPoint3D({ x: 0, y: 0, z: 0 }, { x: 1, y: 1, z: 1 }, 0.5);
    expect(noVisibility.visibility).toBeUndefined();
  });
});

describe('FACE_OVAL_LOOP', () => {
  it('is the canonical MediaPipe oval loop (36 landmarks, closed)', () => {
    expect(FACE_OVAL_LOOP[0]).toBe(10); // forehead
    expect(FACE_OVAL_LOOP).toContain(152); // chin
    expect(FACE_OVAL_LOOP[FACE_OVAL_LOOP.length - 1]).toBe(109);
    expect(FACE_OVAL_LOOP.length).toBe(36);
  });

  it('contains only valid face-mesh indices with no adjacent repeats', () => {
    for (const index of FACE_OVAL_LOOP) {
      expect(index).toBeGreaterThanOrEqual(0);
      expect(index).toBeLessThanOrEqual(467);
    }
    for (let i = 1; i < FACE_OVAL_LOOP.length; i++) {
      expect(FACE_OVAL_LOOP[i]).not.toBe(FACE_OVAL_LOOP[i - 1]);
    }
  });
});

describe('landmark index constants', () => {
  it('uses BlazePose wrists 15/16 and hand wrist 0', () => {
    expect(POSE_WRIST_INDICES).toEqual([15, 16]);
    expect(HAND_WRIST_INDEX).toBe(0);
    expect(HAND_FINGERTIP_INDICES).toEqual([4, 8, 12, 16, 20]);
  });
});

describe('traceFaceOval', () => {
  it('traces one moveTo, ovalLength-1 lineTos and closePath', () => {
    const face = makeSymmetricFace();
    const path = new PathRecorder();

    expect(traceFaceOval(path, face, 100, 100)).toBe(true);
    expect(path.ops).toHaveLength(FACE_OVAL_LOOP.length + 1);
    // Position 0 → x = 1/64 = 0.015625 → 1.5625 px, y = 0.5 → 50 px.
    expect(path.ops[0]).toBe('M 1.5625 50');
    // Position 17 → x = 18/64 → 28.125 px (ops[k] = oval position k).
    expect(path.ops[17]).toBe('L 28.125 50');
    expect(path.ops.at(-1)).toBe('Z');
  });

  it('scales around the oval centroid', () => {
    const face = makeSymmetricFace();
    const path = new PathRecorder();

    // Centroid x = Σ(p+1)·100/64 / 36 = 28.90625; with scale 2: x' = 2x − cx.
    expect(traceFaceOval(path, face, 100, 100, { scale: 2 })).toBe(true);
    expect(path.ops[0]).toBe('M -25.78125 50'); // 2 · 1.5625 − 28.90625
    expect(path.ops[17]).toBe('L 27.34375 50'); // 2 · 28.125 − 28.90625
  });

  it('honors mirroring', () => {
    const face: Point3D[] = Array.from({ length: 478 }, () => ({
      x: 0.5,
      y: 0.5,
      z: 0,
    }));
    face[10] = { x: 0.25, y: 0.5, z: 0 }; // first oval index
    const path = new PathRecorder();

    expect(traceFaceOval(path, face, 100, 100, { mirrored: true })).toBe(true);
    expect(path.ops[0]).toBe('M 75 50'); // (1 - 0.25) * 100
  });

  it('refuses sparse faces without touching the path', () => {
    const path = new PathRecorder();
    expect(traceFaceOval(path, [], 100, 100)).toBe(false);
    expect(traceFaceOval(path, [{ x: 0.5, y: 0.5, z: 0 }], 100, 100)).toBe(false);
    expect(path.ops).toHaveLength(0);
  });
});
