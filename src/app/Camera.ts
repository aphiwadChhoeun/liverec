/**
 * Camera lifecycle: permission, device enumeration and stream management for
 * a single `<video>` element. Switching devices re-requests `getUserMedia`
 * with the new device id; the renderer adapts automatically.
 */

export interface CameraDevice {
  deviceId: string;
  label: string;
}

export interface CameraCallbacks {
  /** Fired whenever the device list changes (hotplug, permission granted). */
  onDevicesChanged?: (devices: readonly CameraDevice[]) => void;
  /** Non-fatal problems (e.g. blocked autoplay) surface here. */
  onError?: (error: Error) => void;
}

/** Map getUserMedia / MediaPipe-adjacent failures to friendly messages. */
export function describeCameraError(error: unknown): string {
  const name = error instanceof Error ? error.name : '';
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return 'Camera permission denied — allow camera access for this site and try again.';
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No usable camera device found (or it does not support the requested size).';
    case 'NotReadableError':
    case 'TrackStartError':
      return 'Camera is busy — close other apps using it and try again.';
    case 'SecurityError':
      return 'Camera access is blocked — the page must be served from HTTPS or localhost.';
    default:
      return error instanceof Error && error.message
        ? error.message
        : 'Unknown camera error.';
  }
}

export class Camera {
  private readonly video: HTMLVideoElement;
  private readonly callbacks: CameraCallbacks;
  private stream: MediaStream | null = null;
  private devices: CameraDevice[] = [];
  private deviceChangeListener: (() => void) | null = null;

  constructor(video: HTMLVideoElement, callbacks: CameraCallbacks = {}) {
    this.video = video;
    this.callbacks = callbacks;
  }

  /** Whether a stream is currently attached. */
  get active(): boolean {
    return this.stream !== null;
  }

  /** Device id of the active track (best effort — may be undefined). */
  get deviceId(): string | undefined {
    const track = this.stream?.getVideoTracks()[0];
    return track?.getSettings().deviceId ?? undefined;
  }

  /**
   * Best guess whether the active camera faces the user. Desktop webcams have
   * no facing mode and default to `true` (mirror selfie view).
   */
  mirroringHint(): boolean {
    const facing = this.stream?.getVideoTracks()[0]?.getSettings().facingMode;
    return facing ? facing === 'user' : true;
  }

  /**
   * Request a camera stream. `deviceId` is optional; omit it for the default
   * user-facing camera. Any existing stream is stopped first.
   */
  async start(deviceId?: string): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        'Camera API unavailable — this page must be served from HTTPS or localhost in a modern browser.',
      );
    }

    this.stopStream();

    const video: MediaTrackConstraints = deviceId
      ? {
          deviceId: { exact: deviceId },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        }
      : {
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 },
        };

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video,
        audio: false,
      });
    } catch (error) {
      throw new Error(describeCameraError(error));
    }

    this.attach();
    this.listenForDeviceChanges();
    // Labels are only exposed after permission has been granted.
    await this.refreshDevices();
  }

  /** Switch to another device mid-stream (renderer keeps running). */
  async switchDevice(deviceId: string): Promise<void> {
    if (!deviceId) return;
    await this.start(deviceId);
  }

  /**
   * List video input devices. Before permission is granted browsers return
   * empty ids/labels, so the UI falls back to a "Default camera" entry.
   */
  async refreshDevices(): Promise<readonly CameraDevice[]> {
    if (!navigator.mediaDevices?.enumerateDevices) return this.devices;
    try {
      const all = await navigator.mediaDevices.enumerateDevices();
      this.devices = all
        .filter((d) => d.kind === 'videoinput')
        .map((d, i) => ({
          deviceId: d.deviceId,
          label: d.label || `Camera ${i + 1}`,
        }));
      this.callbacks.onDevicesChanged?.(this.devices);
    } catch (error) {
      // Enumeration is best-effort; never break the app over it.
      console.warn('[Camera] enumerateDevices failed:', error);
    }
    return this.devices;
  }

  /** Stop all tracks and detach from the video element. */
  stop(): void {
    this.stopStream();
    this.video.srcObject = null;
  }

  /** Full teardown (also removes listeners). */
  dispose(): void {
    this.stop();
    if (this.deviceChangeListener && navigator.mediaDevices) {
      navigator.mediaDevices.removeEventListener(
        'devicechange',
        this.deviceChangeListener,
      );
      this.deviceChangeListener = null;
    }
  }

  private attach(): void {
    if (!this.stream) return;
    this.video.srcObject = this.stream;
    // The element is muted + playsinline so autoplay is policy-safe, but keep
    // a guard in case an embedded context still blocks playback.
    void this.video.play().catch((error: unknown) => {
      this.callbacks.onError?.(
        new Error(
          `Video playback was blocked by the browser: ${
            error instanceof Error ? error.message : String(error)
          } — try clicking the video.`,
        ),
      );
    });
  }

  private stopStream(): void {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
  }

  private listenForDeviceChanges(): void {
    if (this.deviceChangeListener || !navigator.mediaDevices?.addEventListener) {
      return;
    }
    this.deviceChangeListener = () => {
      void this.refreshDevices();
    };
    navigator.mediaDevices.addEventListener(
      'devicechange',
      this.deviceChangeListener,
    );
  }
}
