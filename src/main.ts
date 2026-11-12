/**
 * LiveRec bootstrap: wires camera, detector, renderer, modules and UI together.
 */
import { Camera } from './app/Camera';
import { MediaPipeDetector } from './app/Detector';
import type { DetectorStatus, FeatureId } from './app/Detector';
import { Renderer } from './app/Renderer';
import type { LayerId } from './app/types';
import { SkinSmooth } from './modules/SkinSmooth';
import { ParticleTrail } from './modules/ParticleTrail';
import { clearModules, getModules, registerModule, setModuleEnabled } from './modules/registry';
import { Controls } from './ui/Controls';
import './style.css';

function toMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Which UI layer checkboxes each detector feature backs. */
const FEATURE_LAYERS: Record<FeatureId, readonly LayerId[]> = {
  face: ['faceMesh', 'faceContours'],
  pose: ['pose'],
  hands: ['hands'],
};

function bootstrap(): void {
  const video = document.getElementById('video');
  const canvas = document.getElementById('overlay');
  if (!(video instanceof HTMLVideoElement) || !(canvas instanceof HTMLCanvasElement)) {
    throw new Error('Required #video / #overlay elements are missing.');
  }

  // --- Augmentation modules -------------------------------------------------
  // Add an effect by dropping its file into src/modules and adding one line
  // here. Nothing else in the pipeline needs to change.
  registerModule(new SkinSmooth());
  registerModule(new ParticleTrail());

  const controls = new Controls({
    onStart: () => void start(),
    onStop: stop,
    onCameraChange: (deviceId) => void switchCamera(deviceId),
    onMirrorChange: (mirrored) => {
      mirrorOverride = mirrored;
      renderer.setMirrored(mirrored);
    },
    onLayerToggle: (layer, enabled) => renderer.setLayerEnabled(layer, enabled),
    onModuleToggle: (id, enabled) => {
      if (!setModuleEnabled(id, enabled)) {
        controls.showError(`Could not ${enabled ? 'enable' : 'disable'} module "${id}".`);
      }
    },
    onSnapshot: saveSnapshot,
  });

  const camera = new Camera(video, {
    onDevicesChanged: (devices) => controls.setDevices(devices, camera.deviceId),
    onError: (error) => controls.showError(error.message),
  });

  const detector = new MediaPipeDetector({
    onStatus: (status) => applyDetectorStatus(status),
  });

  const renderer = new Renderer({
    canvas,
    video,
    detector,
    onStats: (stats) => controls.setStats(stats),
    onError: (error) => controls.showError(error.message),
  });

  controls.setModules(getModules());

  let detectorReady = false;
  let starting = false;
  /** User's explicit mirror choice; once set it overrides the camera hint. */
  let mirrorOverride: boolean | null = null;

  async function start(): Promise<void> {
    if (starting || camera.active) return;
    starting = true;
    controls.setBusy(true, 'Starting…');
    controls.clearError();
    try {
      controls.setStatus('Requesting camera permission…');
      await camera.start(controls.selectedDeviceId || undefined);
      controls.setRunning(true);
      syncMirror();

      if (!detectorReady) {
        controls.setStatus('Loading detection models… (first run fetches them from CDN)');
        try {
          await detector.init();
          detectorReady = true;
        } catch (error) {
          // Graceful degradation: keep the plain camera preview running.
          controls.showError(
            `Detection models failed to load: ${toMessage(error)} — ` +
              'camera preview continues without overlays.',
          );
          for (const layers of Object.values(FEATURE_LAYERS)) {
            for (const layer of layers) {
              renderer.setLayerEnabled(layer, false);
              controls.setLayerAvailable(layer, false, 'Detector unavailable');
            }
          }
        }
      }

      applyDetectorStatus(detector.getStatus());
      renderer.start();
    } catch (error) {
      controls.showError(toMessage(error));
      camera.stop();
      controls.setRunning(false);
      controls.setStageMessage('Camera failed — press Start camera to retry.');
      controls.setStatus('Idle.');
    } finally {
      starting = false;
      controls.setBusy(false);
    }
  }

  function stop(): void {
    renderer.dispose();
    camera.stop();
    controls.setRunning(false);
    controls.setStageMessage('Camera stopped — press Start camera to resume.');
    controls.setStatus('Stopped.');
  }

  async function switchCamera(deviceId: string): Promise<void> {
    if (!deviceId) return;
    if (!camera.active) return; // preference picked up by the next start()

    controls.setBusy(true, 'Switching…');
    controls.clearError();
    try {
      await camera.switchDevice(deviceId);
      syncMirror();
      controls.setStatus('Camera switched.');
    } catch (error) {
      controls.showError(toMessage(error));
      if (!camera.active) {
        renderer.dispose();
        controls.setRunning(false);
        controls.setStageMessage('Camera lost — press Start camera to retry.');
      }
      controls.setStatus('Camera switch failed.');
    } finally {
      controls.setBusy(false);
    }
  }

  function saveSnapshot(): void {
    const dataUrl = renderer.captureSnapshot();
    if (!dataUrl) {
      controls.showError('Nothing to capture yet — start the camera first.');
      return;
    }
    const link = document.createElement('a');
    link.href = dataUrl;
    link.download = `liverec-${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
    link.click();
    controls.setStatus('Snapshot saved.');
  }

  function applyDetectorStatus(status: DetectorStatus): void {
    const summary: string[] = [];
    for (const feature of Object.keys(FEATURE_LAYERS) as FeatureId[]) {
      const featureStatus = status[feature];
      for (const layer of FEATURE_LAYERS[feature]) {
        controls.setLayerAvailable(layer, featureStatus.available, featureStatus.error);
      }
      if (featureStatus.available) {
        summary.push(`${feature}: ${featureStatus.delegate ?? '?'}`);
      }
    }
    if (summary.length > 0) {
      controls.setStatus(`Live · ${summary.join(' · ')}`);
    }
  }

  function syncMirror(): void {
    const mirrored = mirrorOverride ?? camera.mirroringHint();
    renderer.setMirrored(mirrored);
    controls.setMirrorChecked(mirrored);
  }

  function shutdown(): void {
    renderer.dispose();
    camera.dispose();
    detector.dispose();
    clearModules();
  }

  window.addEventListener('pagehide', shutdown, { once: true });
}

try {
  bootstrap();
} catch (error) {
  console.error('[LiveRec] fatal bootstrap error:', error);
  const stageText = document.getElementById('stage-msg-text');
  if (stageText) stageText.textContent = `Fatal error: ${toMessage(error)}`;
}
