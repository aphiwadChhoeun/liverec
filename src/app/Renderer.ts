/**
 * Canvas rendering pipeline.
 *
 * Owns the rAF loop and composites, per frame:
 *   1. the video frame (mirrored for the selfie view),
 *   2. the wireframe layers (face tessellation/contours, pose skeleton, hands),
 *   3. every enabled augmentation module, in registration order.
 *
 * Landmarks stay normalized in `DetectionResult`; all pixel conversion (and
 * mirroring) happens here via `ModuleContext` / the geometry helpers.
 *
 * Performance: detection runs on an adaptive schedule — every Nth rendered
 * frame, where N grows while fps is low and shrinks when it recovers — and is
 * additionally skipped while a previous detect call is still in flight.
 * The video preview itself keeps rendering at full rate.
 */
import { FaceLandmarker, HandLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import type { AugmentationModule, ModuleContext } from '../modules/AugmentationModule';
import { getModules } from '../modules/registry';
import { pointVisible, traceFaceOval } from '../utils/geometry';
import type { Detector } from './Detector';
import type { DetectionResult, Edge, LayerId, Point2D, Point3D, Size } from './types';

export interface LayerToggles {
  faceMesh: boolean;
  faceContours: boolean;
  pose: boolean;
  hands: boolean;
}

export interface RenderStats {
  /** Rendered frames per second (exponential moving average). */
  fps: number;
  /** Completed detections per second (EMA). */
  detectFps: number;
  /** Detection currently runs every Nth rendered frame (adaptive). */
  detectEveryN: number;
}

export interface RendererOptions {
  canvas: HTMLCanvasElement;
  video: HTMLVideoElement;
  detector: Detector;
  /** Mirror the preview (selfie view). Default `true`. */
  mirrored?: boolean;
  /** Initially enabled wireframe layers (all default to `true`). */
  layers?: Partial<LayerToggles>;
  /** Stats callback, throttled to roughly 2×/second. */
  onStats?: (stats: RenderStats) => void;
  /** Surfaces module / detection errors so the UI can show them. */
  onError?: (error: Error) => void;
}

/** Result shape handed to modules before the first detection completes. */
const EMPTY_RESULT: DetectionResult = {
  timestamp: 0,
  videoSize: { width: 0, height: 0 },
  face: { landmarks: [] },
  pose: { landmarks: [] },
  hands: { landmarks: [], handedness: [] },
};

const LAYER_STYLE = {
  faceMesh: { color: 'rgba(56, 189, 248, 0.22)', width: 1 },
  faceContours: { color: 'rgba(125, 211, 252, 0.85)', width: 1.5 },
  pose: {
    color: 'rgba(74, 222, 128, 0.8)',
    width: 4,
    joint: 'rgba(74, 222, 128, 0.9)',
    jointRadius: 5,
  },
  hands: {
    color: 'rgba(251, 191, 36, 0.9)',
    width: 3,
    joint: 'rgba(251, 191, 36, 0.95)',
    jointRadius: 3.5,
  },
} as const;

/** Adaptive-detection thresholds. */
const FPS_LOW = 22;
const FPS_HIGH = 28;
const MAX_DETECT_SKIP = 4;
/** Frames between adaptive adjustments (~1.5 s at 60 fps). */
const ADJUST_INTERVAL_FRAMES = 90;
/** Frames between stats emissions (~0.5 s at 60 fps). */
const STATS_INTERVAL_FRAMES = 30;
const STAGE_BACKGROUND = '#05080f';

export class Renderer {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly video: HTMLVideoElement;
  private readonly detector: Detector;
  private readonly layers: LayerToggles;
  private readonly onStats?: (stats: RenderStats) => void;
  private readonly onError?: (error: Error) => void;

  private mirrored: boolean;
  private readonly moduleCtx: ModuleContext;
  /** Backing object for `moduleCtx.canvasSize` — mutated, never reallocated. */
  private readonly moduleCanvasSize: Size = { width: 0, height: 0 };

  private rafId = 0;
  private running = false;
  private startTime = 0;
  private lastFrameTime = 0;
  private lastResult: DetectionResult | null = null;

  private frameIndex = 0;
  private detectBusy = false;
  private detectEveryN = 1;
  private adjustCounter = 0;
  private statsCounter = 0;
  private fpsEma = 0;
  private detectFpsEma = 0;
  private lastDetectCompletedAt = 0;

  constructor(options: RendererOptions) {
    this.canvas = options.canvas;
    const ctx = this.canvas.getContext('2d');
    if (!ctx) {
      throw new Error('Canvas 2D context is unavailable in this browser.');
    }
    this.ctx = ctx;
    this.video = options.video;
    this.detector = options.detector;
    this.mirrored = options.mirrored ?? true;
    this.layers = {
      faceMesh: true,
      faceContours: true,
      pose: true,
      hands: true,
      ...options.layers,
    };
    this.onStats = options.onStats;
    this.onError = options.onError;

    this.moduleCtx = this.createModuleContext();
  }

  get isRunning(): boolean {
    return this.running;
  }

  get isMirrored(): boolean {
    return this.mirrored;
  }

  setMirrored(mirrored: boolean): void {
    this.mirrored = mirrored;
  }

  isLayerEnabled(layer: LayerId): boolean {
    return this.layers[layer];
  }

  /** Toggling takes effect on the next frame — no restart needed. */
  setLayerEnabled(layer: LayerId, enabled: boolean): void {
    this.layers[layer] = enabled;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.lastResult = null;
    this.frameIndex = 0;
    this.adjustCounter = 0;
    this.statsCounter = 0;
    this.fpsEma = 0;
    this.detectFpsEma = 0;
    this.lastDetectCompletedAt = 0;
    this.startTime = performance.now();
    this.lastFrameTime = this.startTime;
    this.rafId = requestAnimationFrame(this.loop);
  }

  stop(): void {
    if (!this.running) return;
    this.running = false;
    cancelAnimationFrame(this.rafId);
  }

  /** Stop and blank the canvas (used when the camera stops). */
  dispose(): void {
    this.stop();
    this.clearCanvas();
  }

  /** Blank the stage so a stopped camera does not show a stale frame. */
  clearCanvas(): void {
    const { width, height } = this.canvas;
    if (width === 0 || height === 0) return;
    this.ctx.save();
    this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    this.ctx.fillStyle = STAGE_BACKGROUND;
    this.ctx.fillRect(0, 0, width, height);
    this.ctx.restore();
  }

  /** PNG data URL of the current composite, or `null` when nothing is drawn. */
  captureSnapshot(): string | null {
    if (this.canvas.width === 0 || this.canvas.height === 0) return null;
    try {
      return this.canvas.toDataURL('image/png');
    } catch (error) {
      console.warn('[Renderer] snapshot failed:', error);
      return null;
    }
  }

  // -----------------------------------------------------------------------
  // render loop
  // -----------------------------------------------------------------------

  private readonly loop = (now: number): void => {
    if (!this.running) return;
    this.rafId = requestAnimationFrame(this.loop);

    // No decodable frame yet (camera still warming up / switching) — keep the
    // loop alive but skip everything.
    if (this.video.readyState < 2 || this.video.videoWidth === 0) return;

    const dt = Math.max(0.001, (now - this.lastFrameTime) / 1000);
    this.lastFrameTime = now;
    const time = (now - this.startTime) / 1000;
    this.updateFps(dt);

    this.ensureCanvasSize();
    this.drawVideoFrame();
    this.frameIndex++;

    this.maybeDetect();
    const result = this.lastResult ?? EMPTY_RESULT;
    // `EMPTY_RESULT` carries no detections, so the wireframe pass would be a
    // no-op; skip it until the first real result lands.
    if (this.lastResult) {
      this.drawWireframe(result);
    }

    this.runModules(result, time, dt);
    this.adaptDetectionRate();
    this.maybeEmitStats();
  };

  private updateFps(dt: number): void {
    const instant = 1 / dt;
    this.fpsEma =
      this.fpsEma === 0 ? instant : this.fpsEma + (instant - this.fpsEma) * 0.1;
  }

  /** Match the canvas backing store to the video resolution (handles camera switches). */
  private ensureCanvasSize(): void {
    const { videoWidth, videoHeight } = this.video;
    if (this.canvas.width !== videoWidth || this.canvas.height !== videoHeight) {
      this.canvas.width = videoWidth;
      this.canvas.height = videoHeight;
      // Resetting the width clears ctx state — we set everything per frame,
      // but the transform must be neutral for the module paths.
      this.ctx.setTransform(1, 0, 0, 1, 0, 0);
    }
    this.moduleCanvasSize.width = this.canvas.width;
    this.moduleCanvasSize.height = this.canvas.height;
  }

  private drawVideoFrame(): void {
    const { width, height } = this.canvas;
    this.ctx.save();
    if (this.mirrored) {
      this.ctx.translate(width, 0);
      this.ctx.scale(-1, 1);
    }
    this.ctx.drawImage(this.video, 0, 0, width, height);
    this.ctx.restore();
  }

  private maybeDetect(): void {
    if (this.detectBusy) return;
    if (this.frameIndex % this.detectEveryN !== 0) return;

    this.detectBusy = true;
    void this.detector
      .detect(this.video, performance.now())
      .then((result) => {
        if (!result) return;
        this.lastResult = result;
        const completedAt = performance.now();
        if (this.lastDetectCompletedAt > 0) {
          const instant =
            1000 / Math.max(1, completedAt - this.lastDetectCompletedAt);
          this.detectFpsEma =
            this.detectFpsEma === 0
              ? instant
              : this.detectFpsEma + (instant - this.detectFpsEma) * 0.15;
        }
        this.lastDetectCompletedAt = completedAt;
      })
      .catch((error: unknown) => {
        this.onError?.(
          error instanceof Error ? error : new Error(String(error)),
        );
      })
      .finally(() => {
        this.detectBusy = false;
      });
  }

  /** Nudge detection frequency based on rendered fps (hysteresis, 1–4). */
  private adaptDetectionRate(): void {
    if (++this.adjustCounter < ADJUST_INTERVAL_FRAMES) return;
    this.adjustCounter = 0;
    if (this.fpsEma < FPS_LOW && this.detectEveryN < MAX_DETECT_SKIP) {
      this.detectEveryN++;
    } else if (this.fpsEma > FPS_HIGH && this.detectEveryN > 1) {
      this.detectEveryN--;
    }
  }

  private maybeEmitStats(): void {
    if (++this.statsCounter < STATS_INTERVAL_FRAMES) return;
    this.statsCounter = 0;
    this.onStats?.({
      fps: Math.round(this.fpsEma * 10) / 10,
      detectFps: Math.round(this.detectFpsEma * 10) / 10,
      detectEveryN: this.detectEveryN,
    });
  }

  // -----------------------------------------------------------------------
  // wireframe layers
  // -----------------------------------------------------------------------

  private drawWireframe(result: DetectionResult): void {
    if (this.layers.faceMesh) {
      for (const face of result.face?.landmarks ?? []) {
        this.drawConnections(
          face,
          FaceLandmarker.FACE_LANDMARKS_TESSELATION,
          LAYER_STYLE.faceMesh.color,
          LAYER_STYLE.faceMesh.width,
          0,
        );
      }
    }
    if (this.layers.faceContours) {
      for (const face of result.face?.landmarks ?? []) {
        this.drawConnections(
          face,
          FaceLandmarker.FACE_LANDMARKS_CONTOURS,
          LAYER_STYLE.faceContours.color,
          LAYER_STYLE.faceContours.width,
          0,
        );
      }
    }
    if (this.layers.pose) {
      for (const pose of result.pose?.landmarks ?? []) {
        this.drawConnections(
          pose,
          PoseLandmarker.POSE_CONNECTIONS,
          LAYER_STYLE.pose.color,
          LAYER_STYLE.pose.width,
          0.5,
        );
        this.drawJoints(
          pose,
          LAYER_STYLE.pose.joint,
          LAYER_STYLE.pose.jointRadius,
          0.5,
        );
      }
    }
    if (this.layers.hands) {
      for (const hand of result.hands?.landmarks ?? []) {
        this.drawConnections(
          hand,
          HandLandmarker.HAND_CONNECTIONS,
          LAYER_STYLE.hands.color,
          LAYER_STYLE.hands.width,
          0,
        );
        this.drawJoints(
          hand,
          LAYER_STYLE.hands.joint,
          LAYER_STYLE.hands.jointRadius,
          0,
        );
      }
    }
  }

  /**
   * Stroke an edge list. Visibility gating is opt-in via `minVisibility > 0`
   * (face/hand landmarks carry no meaningful visibility, pose landmarks do).
   */
  private drawConnections(
    points: readonly Point3D[],
    connections: readonly Edge[],
    color: string,
    lineWidth: number,
    minVisibility: number,
  ): void {
    const ctx = this.ctx;
    const { width: w, height: h } = this.canvas;
    const mirrored = this.mirrored;

    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';
    ctx.beginPath();
    for (const edge of connections) {
      const a = points[edge.start];
      const b = points[edge.end];
      if (!a || !b) continue;
      if (minVisibility > 0 && (!pointVisible(a, minVisibility) || !pointVisible(b, minVisibility))) {
        continue;
      }
      ctx.moveTo(mirrored ? (1 - a.x) * w : a.x * w, a.y * h);
      ctx.lineTo(mirrored ? (1 - b.x) * w : b.x * w, b.y * h);
    }
    ctx.stroke();
  }

  private drawJoints(
    points: readonly Point3D[],
    color: string,
    radius: number,
    minVisibility: number,
  ): void {
    const ctx = this.ctx;
    const { width: w, height: h } = this.canvas;
    const mirrored = this.mirrored;

    ctx.fillStyle = color;
    ctx.beginPath();
    for (const p of points) {
      if (minVisibility > 0 && !pointVisible(p, minVisibility)) continue;
      const x = mirrored ? (1 - p.x) * w : p.x * w;
      const y = p.y * h;
      // moveTo before each arc prevents connector lines between circles.
      ctx.moveTo(x + radius, y);
      ctx.arc(x, y, radius, 0, Math.PI * 2);
    }
    ctx.fill();
  }

  // -----------------------------------------------------------------------
  // augmentation modules
  // -----------------------------------------------------------------------

  private runModules(result: DetectionResult, time: number, dt: number): void {
    const modules = getModules();
    if (modules.length === 0) return;

    const ctx = this.moduleCtx;
    ctx.mirrored = this.mirrored;
    ctx.time = time;
    ctx.dt = dt;

    for (const module of modules) {
      if (!module.enabled) continue;
      try {
        module.update(this.ctx, result, ctx);
      } catch (error) {
        this.handleModuleError(module, error);
      }
    }
  }

  /**
   * A module that throws must not kill the render loop: log, surface, and
   * disable it for the rest of the session.
   */
  private handleModuleError(module: AugmentationModule, error: unknown): void {
    console.error(`[Renderer] module "${module.id}" threw — disabling it:`, error);
    module.enabled = false;
    try {
      module.onDisable?.();
    } catch (secondary) {
      console.error(`[Renderer] onDisable of "${module.id}" also threw:`, secondary);
    }
    const message = error instanceof Error ? error.message : String(error);
    this.onError?.(
      new Error(`Module "${module.name}" failed and was disabled: ${message}`),
    );
  }

  private createModuleContext(): ModuleContext {
    const renderer = this;
    const canvasSize = this.moduleCanvasSize;
    const toCanvas = (point: Point3D): Point2D => ({
      x: (renderer.mirrored ? 1 - point.x : point.x) * canvasSize.width,
      y: point.y * canvasSize.height,
    });

    return {
      video: this.video,
      mirrored: this.mirrored,
      canvasSize,
      time: 0,
      dt: 0,
      toCanvas,
      landmarkToPixel: toCanvas,
      faceOvalPath: (faceIndex = 0, opts = {}) => {
        const face = renderer.lastResult?.face?.landmarks[faceIndex];
        if (!face) return null;
        const path = new Path2D();
        const traced = traceFaceOval(
          path,
          face,
          canvasSize.width,
          canvasSize.height,
          { mirrored: renderer.mirrored, scale: opts.scale },
        );
        return traced ? path : null;
      },
    };
  }
}
