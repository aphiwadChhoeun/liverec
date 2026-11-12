/**
 * The augmentation-module contract.
 *
 * A module is a self-contained visual effect that receives the rendered
 * canvas context, the latest detection result and a `ModuleContext` of
 * pixel-space helpers once per frame — and draws whatever it likes on top.
 * The core pipeline (camera → detector → renderer) never changes when modules
 * are added or removed; see `src/modules/README.md` for the full guide.
 */
import type { DetectionResult, Point2D, Point3D, Size } from '../app/types';

/**
 * Per-frame context handed to modules. Provides pixel-space helpers so a
 * module never has to re-implement mirroring or video-size math itself.
 */
export interface ModuleContext {
  /** Source video the canvas is compositing this frame. */
  video: HTMLVideoElement;
  /** Whether the canvas shows a mirrored (selfie) view. */
  mirrored: boolean;
  /** Current canvas backing-store size in device pixels. */
  canvasSize: Size;
  /** Seconds since the renderer started (for time-based effects). */
  time: number;
  /** Seconds since the previous rendered frame (frame-rate independent motion). */
  dt: number;
  /** Convert a normalized landmark to canvas pixels (honors mirroring). */
  toCanvas(point: Point3D): Point2D;
  /** Alias of `toCanvas`, kept for discoverability. */
  landmarkToPixel(point: Point3D): Point2D;
  /**
   * `Path2D` tracing the face oval of the requested face in canvas pixels,
   * or `null` when no face is tracked.
   *
   * @param faceIndex Index into `result.face.landmarks` (default 0).
   * @param opts.scale Expand the oval around its centroid (`1` = exact,
   *   `1.1` = 10% padding). Default `1`.
   */
  faceOvalPath(faceIndex?: number, opts?: { scale?: number }): Path2D | null;
}

export interface AugmentationModule {
  /** Stable unique id (used by the registry and the UI). */
  id: string;
  /** Human-readable name shown in the UI. */
  name: string;
  /** One-line description shown under the checkbox. */
  description?: string;
  /** Toggled at runtime — disabled modules are skipped entirely. */
  enabled: boolean;
  /**
   * Called once per rendered frame, after the video frame and wireframe
   * layers, in registration order. Wrap the ctx in save/restore when
   * changing state — the renderer does not reset it between modules.
   */
  update(
    ctx: CanvasRenderingContext2D,
    result: DetectionResult,
    context: ModuleContext,
  ): void;
  /** Optional lifecycle hooks. */
  onEnable?(): void;
  onDisable?(): void;
  dispose?(): void;
}
