/**
 * Detection backend contract + the default MediaPipe Tasks implementation.
 *
 * `Detector` is deliberately narrow so the backend can be swapped (a remote
 * WebSocket detector, a TensorFlow.js port, …) without touching the renderer:
 * init once, then feed it the live `<video>` and a monotonic timestamp, get
 * back a normalized `DetectionResult`.
 */
import {
  FaceLandmarker,
  FilesetResolver,
  HandLandmarker,
  PoseLandmarker,
} from '@mediapipe/tasks-vision';
import type {
  Classifications,
  NormalizedLandmark,
} from '@mediapipe/tasks-vision';
import pkg from '../../package.json';
import type {
  Blendshape,
  DetectionResult,
  Point3D,
} from './types';

/** Swappable detection backend. */
export interface Detector {
  /** Load models / prepare resources. Must be idempotent (retryable). */
  init(): Promise<void>;
  /**
   * Run detection for the current video frame.
   *
   * @param now Monotonic timestamp in ms (e.g. `performance.now()`); backends
   *   that need strictly increasing timestamps get that guarantee here.
   * @returns The normalized detection, or `null` to skip this frame (busy /
   *   no video yet / nothing initialized).
   */
  detect(video: HTMLVideoElement, now: number): Promise<DetectionResult | null>;
  /** Release resources. */
  dispose(): void;
}

// ---------------------------------------------------------------------------
// MediaPipe implementation
// ---------------------------------------------------------------------------

export type FeatureId = 'face' | 'pose' | 'hands';
export type DelegateId = 'CPU' | 'GPU';

export interface FeatureStatus {
  /** Whether the feature was requested. */
  requested: boolean;
  /** Whether a landmarker instance is ready. */
  available: boolean;
  /** Compute delegate the landmarker ended up on. */
  delegate: DelegateId | null;
  /** Human-readable failure reason when `available` is false. */
  error?: string;
}

export type DetectorStatus = Readonly<Record<FeatureId, FeatureStatus>>;

export interface MediaPipeDetectorOptions {
  /** Which landmarkers to enable (default: all three). */
  features?: Partial<Record<FeatureId, boolean>>;
  /** Override model asset URLs (e.g. to self-host the `.task` files). */
  modelUrls?: Partial<Record<FeatureId, string>>;
  /** Override the WASM fileset base path (e.g. to serve the wasm locally). */
  wasmBasePath?: string;
  numFaces?: number;
  numPoses?: number;
  numHands?: number;
  /** Notified after init (and any future status change). */
  onStatus?: (status: DetectorStatus) => void;
}

const MODEL_CDN = 'https://storage.googleapis.com/mediapipe-models';

const DEFAULT_MODEL_URLS: Record<FeatureId, string> = {
  face: `${MODEL_CDN}/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`,
  pose: `${MODEL_CDN}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task`,
  hands: `${MODEL_CDN}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`,
};

// The WASM runtime must match the installed npm package, so the version is
// read straight from package.json (both are pinned there).
const TASKS_VISION_VERSION = pkg.dependencies['@mediapipe/tasks-vision'];
const DEFAULT_WASM_BASE_PATH = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VISION_VERSION}/wasm`;

/** `WasmFileset` is not exported by the package; derive it from the resolver. */
type VisionFileset = Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>;

function toErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return String(error);
}

/** Map MediaPipe's per-face classifications onto plain blendshape records. */
function toBlendshapes(classifications: Classifications): Blendshape[] {
  return classifications.categories.map((category) => ({
    categoryName: category.categoryName,
    displayName: category.displayName || undefined,
    score: category.score,
  }));
}

/** Copy normalized landmarks into our plain `Point3D` shape. */
function toPoint3Ds(landmarks: NormalizedLandmark[]): Point3D[] {
  return landmarks.map((lm) => ({
    x: lm.x,
    y: lm.y,
    z: lm.z,
    visibility: lm.visibility,
  }));
}

export class MediaPipeDetector implements Detector {
  private readonly features: Record<FeatureId, boolean>;
  private readonly modelUrls: Record<FeatureId, string>;
  private readonly wasmBasePath: string;
  private readonly numFaces: number;
  private readonly numPoses: number;
  private readonly numHands: number;
  private readonly onStatus?: (status: DetectorStatus) => void;

  private fileset: VisionFileset | null = null;
  private face: FaceLandmarker | null = null;
  private pose: PoseLandmarker | null = null;
  private hands: HandLandmarker | null = null;
  private readonly status: Record<FeatureId, FeatureStatus>;

  private busy = false;
  private lastTimestamp = 0;
  private initPromise: Promise<void> | null = null;

  constructor(options: MediaPipeDetectorOptions = {}) {
    this.features = { face: true, pose: true, hands: true, ...options.features };
    this.modelUrls = { ...DEFAULT_MODEL_URLS, ...options.modelUrls };
    this.wasmBasePath = options.wasmBasePath ?? DEFAULT_WASM_BASE_PATH;
    this.numFaces = options.numFaces ?? 1;
    this.numPoses = options.numPoses ?? 1;
    this.numHands = options.numHands ?? 2;
    this.onStatus = options.onStatus;
    this.status = {
      face: { requested: this.features.face, available: false, delegate: null },
      pose: { requested: this.features.pose, available: false, delegate: null },
      hands: { requested: this.features.hands, available: false, delegate: null },
    };
  }

  async init(): Promise<void> {
    // Idempotent; a failed init clears the cached promise so it can be retried.
    if (!this.initPromise) {
      this.initPromise = this.doInit().catch((error: unknown) => {
        this.initPromise = null;
        throw error;
      });
    }
    return this.initPromise;
  }

  private async doInit(): Promise<void> {
    this.fileset = await FilesetResolver.forVisionTasks(this.wasmBasePath);
    // Each landmarker degrades independently — a failed model disables only
    // its own feature, never the whole detector.
    await Promise.all([this.initFace(), this.initPose(), this.initHands()]);
    this.emitStatus();
  }

  private async initFace(): Promise<void> {
    if (!this.features.face) return;
    try {
      const { instance, delegate } = await this.withFallback((delegate) =>
        FaceLandmarker.createFromOptions(this.requireFileset(), {
          baseOptions: { modelAssetPath: this.modelUrls.face, delegate },
          runningMode: 'VIDEO',
          numFaces: this.numFaces,
          outputFaceBlendshapes: true,
        }),
      );
      this.face = instance;
      this.status.face = { requested: true, available: true, delegate };
    } catch (error) {
      console.warn('[MediaPipeDetector] face landmarker unavailable:', error);
      this.status.face = {
        requested: true,
        available: false,
        delegate: null,
        error: toErrorMessage(error),
      };
    }
  }

  private async initPose(): Promise<void> {
    if (!this.features.pose) return;
    try {
      const { instance, delegate } = await this.withFallback((delegate) =>
        PoseLandmarker.createFromOptions(this.requireFileset(), {
          baseOptions: { modelAssetPath: this.modelUrls.pose, delegate },
          runningMode: 'VIDEO',
          numPoses: this.numPoses,
        }),
      );
      this.pose = instance;
      this.status.pose = { requested: true, available: true, delegate };
    } catch (error) {
      console.warn('[MediaPipeDetector] pose landmarker unavailable:', error);
      this.status.pose = {
        requested: true,
        available: false,
        delegate: null,
        error: toErrorMessage(error),
      };
    }
  }

  private async initHands(): Promise<void> {
    if (!this.features.hands) return;
    try {
      const { instance, delegate } = await this.withFallback((delegate) =>
        HandLandmarker.createFromOptions(this.requireFileset(), {
          baseOptions: { modelAssetPath: this.modelUrls.hands, delegate },
          runningMode: 'VIDEO',
          numHands: this.numHands,
        }),
      );
      this.hands = instance;
      this.status.hands = { requested: true, available: true, delegate };
    } catch (error) {
      console.warn('[MediaPipeDetector] hand landmarker unavailable:', error);
      this.status.hands = {
        requested: true,
        available: false,
        delegate: null,
        error: toErrorMessage(error),
      };
    }
  }

  async detect(
    video: HTMLVideoElement,
    now: number,
  ): Promise<DetectionResult | null> {
    // Busy flag: skip the frame while a previous pass is still in flight
    // (relevant for future async backends; MediaPipe's call is synchronous).
    if (this.busy) return null;
    if (!this.face && !this.pose && !this.hands) return null;
    if (video.readyState < 2 || video.videoWidth === 0) return null;

    this.busy = true;
    try {
      // MediaPipe task instances require strictly increasing timestamps.
      const timestamp = now > this.lastTimestamp ? now : this.lastTimestamp + 1;
      this.lastTimestamp = timestamp;

      const result: DetectionResult = {
        timestamp,
        videoSize: { width: video.videoWidth, height: video.videoHeight },
      };

      if (this.face) {
        const r = this.face.detectForVideo(video, timestamp);
        result.face = {
          landmarks: r.faceLandmarks.map(toPoint3Ds),
          blendshapes: r.faceBlendshapes.map(toBlendshapes),
        };
      }
      if (this.pose) {
        const r = this.pose.detectForVideo(video, timestamp);
        result.pose = {
          landmarks: r.landmarks.map(toPoint3Ds),
          worldLandmarks: r.worldLandmarks.map(toPoint3Ds),
        };
      }
      if (this.hands) {
        const r = this.hands.detectForVideo(video, timestamp);
        result.hands = {
          landmarks: r.landmarks.map(toPoint3Ds),
          handedness: r.handedness.map(
            (categories) => categories[0]?.categoryName ?? 'Unknown',
          ),
        };
      }
      return result;
    } finally {
      this.busy = false;
    }
  }

  dispose(): void {
    this.face?.close();
    this.pose?.close();
    this.hands?.close();
    this.face = null;
    this.pose = null;
    this.hands = null;
    this.fileset = null;
    this.initPromise = null;
    this.lastTimestamp = 0;
  }

  /** Snapshot of per-feature availability (safe to hand to the UI). */
  getStatus(): DetectorStatus {
    return {
      face: { ...this.status.face },
      pose: { ...this.status.pose },
      hands: { ...this.status.hands },
    };
  }

  /**
   * Tries the GPU delegate first and falls back to CPU when the GPU path is
   * unavailable (no WebGL / blocked drivers). Re-throws when both fail.
   */
  private async withFallback<T>(
    create: (delegate: DelegateId) => Promise<T>,
  ): Promise<{ instance: T; delegate: DelegateId }> {
    try {
      return { instance: await create('GPU'), delegate: 'GPU' };
    } catch (gpuError) {
      console.warn(
        '[MediaPipeDetector] GPU delegate failed, retrying on CPU:',
        gpuError,
      );
      return { instance: await create('CPU'), delegate: 'CPU' };
    }
  }

  private requireFileset(): VisionFileset {
    if (!this.fileset) {
      throw new Error('Detector not initialized — call init() first.');
    }
    return this.fileset;
  }

  private emitStatus(): void {
    this.onStatus?.(this.getStatus());
  }
}
