/**
 * Shared types for the detection pipeline and rendering layers.
 *
 * Landmark coordinates are **normalized** [0,1] relative to the un-mirrored
 * video frame (as produced by MediaPipe). Pixel-space conversion — including
 * optional mirroring for the selfie view — happens only in the renderer /
 * `ModuleContext` helpers, never inside a detector.
 */

export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D {
  x: number;
  y: number;
  z: number;
  /** Likelihood the landmark is visible in frame (0–1), when the backend provides it. */
  visibility?: number;
}

export interface Size {
  width: number;
  height: number;
}

/** A named blendshape score (e.g. `eyeBlinkLeft`, `mouthSmileRight`). */
export interface Blendshape {
  categoryName: string;
  displayName?: string;
  score: number;
}

export interface FaceDetection {
  /** One landmark set (478 points) per detected face. */
  landmarks: Point3D[][];
  /** Blendshape scores per detected face, same order as `landmarks`. */
  blendshapes?: Blendshape[][];
}

export interface PoseDetection {
  /** One landmark set (33 points) per detected person. */
  landmarks: Point3D[][];
  /** Same landmarks in world (metric) coordinates, per person. */
  worldLandmarks?: Point3D[][];
}

export interface HandDetection {
  /** One landmark set (21 points) per detected hand. */
  landmarks: Point3D[][];
  /** Handedness label per detected hand ("Left" / "Right"). */
  handedness: string[];
}

export interface DetectionResult {
  /** Monotonic timestamp (ms) the detection ran at. */
  timestamp: number;
  face?: FaceDetection;
  pose?: PoseDetection;
  hands?: HandDetection;
  videoSize: Size;
  /**
   * Reserved extension point for backends that also produce segmentation
   * masks (e.g. person segmentation for background replacement). The default
   * MediaPipe detector does not populate it yet; the payload is intentionally
   * opaque so a future detector can pass its native mask representation
   * through to modules without breaking this contract.
   */
  segmentation?: unknown;
}

/** Undirected landmark edge (structurally identical to MediaPipe's `Connection`). */
export interface Edge {
  start: number;
  end: number;
}

/** Toggleable wireframe layers, mirrored by the UI checkboxes. */
export type LayerId = 'faceMesh' | 'faceContours' | 'pose' | 'hands';
