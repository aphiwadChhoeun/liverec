/**
 * Thin UI layer: wires the static controls from `index.html` to callbacks and
 * renders the dynamic lists (camera devices, augmentation modules). Keeps all
 * DOM manipulation in one place; everything else talks to callbacks/setters.
 */
import type { CameraDevice } from '../app/Camera';
import type { RenderStats } from '../app/Renderer';
import type { LayerId } from '../app/types';
import type { AugmentationModule } from '../modules/AugmentationModule';

export interface ControlsHandlers {
  onStart(): void;
  onStop(): void;
  onCameraChange(deviceId: string): void;
  onMirrorChange(mirrored: boolean): void;
  onLayerToggle(layer: LayerId, enabled: boolean): void;
  onModuleToggle(moduleId: string, enabled: boolean): void;
  onSnapshot(): void;
}

const LAYER_INPUT_IDS: Record<LayerId, string> = {
  faceMesh: 'layer-faceMesh',
  faceContours: 'layer-faceContours',
  pose: 'layer-pose',
  hands: 'layer-hands',
};

function requireElement<T extends HTMLElement>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing required element: ${selector}`);
  }
  return element;
}

export class Controls {
  private readonly handlers: ControlsHandlers;
  private readonly startButton: HTMLButtonElement;
  private readonly snapshotButton: HTMLButtonElement;
  private readonly cameraSelect: HTMLSelectElement;
  private readonly mirrorToggle: HTMLInputElement;
  private readonly layerInputs: Record<LayerId, HTMLInputElement>;
  private readonly modulesList: HTMLElement;
  private readonly fpsBadge: HTMLElement;
  private readonly detectBadge: HTMLElement;
  private readonly stageMsg: HTMLElement;
  private readonly stageMsgText: HTMLElement;
  private readonly statusLine: HTMLElement;
  private readonly errorLine: HTMLElement;

  private running = false;

  constructor(handlers: ControlsHandlers) {
    this.handlers = handlers;
    this.startButton = requireElement<HTMLButtonElement>('#btn-start');
    this.snapshotButton = requireElement<HTMLButtonElement>('#btn-snapshot');
    this.cameraSelect = requireElement<HTMLSelectElement>('#camera-select');
    this.mirrorToggle = requireElement<HTMLInputElement>('#mirror-toggle');
    this.modulesList = requireElement<HTMLElement>('#modules-list');
    this.fpsBadge = requireElement<HTMLElement>('#fps-badge');
    this.detectBadge = requireElement<HTMLElement>('#detect-badge');
    this.stageMsg = requireElement<HTMLElement>('#stage-msg');
    this.stageMsgText = requireElement<HTMLElement>('#stage-msg-text');
    this.statusLine = requireElement<HTMLElement>('#status-line');
    this.errorLine = requireElement<HTMLElement>('#error-line');

    this.layerInputs = {
      faceMesh: requireElement<HTMLInputElement>(`#${LAYER_INPUT_IDS.faceMesh}`),
      faceContours: requireElement<HTMLInputElement>(
        `#${LAYER_INPUT_IDS.faceContours}`,
      ),
      pose: requireElement<HTMLInputElement>(`#${LAYER_INPUT_IDS.pose}`),
      hands: requireElement<HTMLInputElement>(`#${LAYER_INPUT_IDS.hands}`),
    };

    this.startButton.addEventListener('click', () => {
      if (this.running) {
        this.handlers.onStop();
      } else {
        this.handlers.onStart();
      }
    });
    this.snapshotButton.addEventListener('click', () => this.handlers.onSnapshot());
    this.cameraSelect.addEventListener('change', () => {
      this.handlers.onCameraChange(this.cameraSelect.value);
    });
    this.mirrorToggle.addEventListener('change', () => {
      this.handlers.onMirrorChange(this.mirrorToggle.checked);
    });
    for (const layer of Object.keys(LAYER_INPUT_IDS) as LayerId[]) {
      this.layerInputs[layer].addEventListener('change', () => {
        this.handlers.onLayerToggle(layer, this.layerInputs[layer].checked);
      });
    }
  }

  /** Device id currently chosen in the dropdown ('' = default camera). */
  get selectedDeviceId(): string {
    return this.cameraSelect.value;
  }

  /** Reflect an external mirror change (camera switch) in the checkbox. */
  setMirrorChecked(checked: boolean): void {
    this.mirrorToggle.checked = checked;
  }

  /** Disable the start button while async work is in flight. */
  setBusy(busy: boolean, label?: string): void {
    this.startButton.disabled = busy;
    if (label) this.startButton.textContent = label;
  }

  setRunning(running: boolean): void {
    this.running = running;
    this.startButton.textContent = running ? 'Stop camera' : 'Start camera';
    this.startButton.setAttribute('aria-pressed', String(running));
    this.snapshotButton.disabled = !running;
    this.setStageMessage(
      running ? null : 'Press Start camera to begin.',
    );
  }

  setStageMessage(text: string | null): void {
    if (text === null) {
      this.stageMsg.hidden = true;
      return;
    }
    this.stageMsgText.textContent = text;
    this.stageMsg.hidden = false;
  }

  /** Rebuild the camera dropdown. `selectedId` (if any) is preserved. */
  setDevices(devices: readonly CameraDevice[], selectedId?: string): void {
    const previous = selectedId ?? this.cameraSelect.value;

    const defaultOption = document.createElement('option');
    defaultOption.value = '';
    defaultOption.textContent = 'Default camera';
    this.cameraSelect.replaceChildren(defaultOption);

    for (const device of devices) {
      if (!device.deviceId) continue; // pre-permission placeholders
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.textContent = device.label;
      this.cameraSelect.append(option);
    }

    const match = Array.from(this.cameraSelect.options).find(
      (option) => option.value === previous,
    );
    this.cameraSelect.value = match ? previous : '';
  }

  /** Render one checkbox per registered module. */
  setModules(modules: readonly AugmentationModule[]): void {
    this.modulesList.replaceChildren();
    for (const module of modules) {
      const label = document.createElement('label');
      label.className = 'check';

      const input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = module.enabled;
      input.id = `module-${module.id}`;
      input.setAttribute('aria-label', module.name);
      input.addEventListener('change', () => {
        this.handlers.onModuleToggle(module.id, input.checked);
      });

      const text = document.createElement('span');
      const name = document.createElement('span');
      name.textContent = module.name;
      text.append(name);
      if (module.description) {
        const description = document.createElement('small');
        description.textContent = module.description;
        text.append(description);
      }

      label.append(input, text);
      this.modulesList.append(label);
    }
  }

  /**
   * Grey out a layer checkbox when its detector feature is unavailable
   * (model failed to load), with the reason as a tooltip.
   */
  setLayerAvailable(layer: LayerId, available: boolean, reason?: string): void {
    const input = this.layerInputs[layer];
    input.disabled = !available;
    if (!available) input.checked = false;
    const label = input.closest('label');
    if (label) label.title = available ? '' : (reason ?? 'Unavailable');
  }

  setStats(stats: RenderStats): void {
    this.fpsBadge.textContent = `${stats.fps.toFixed(1)} fps`;
    this.fpsBadge.classList.toggle('low', stats.fps < 24);
    this.detectBadge.textContent = `detect ${stats.detectFps.toFixed(0)}/s`;
    this.detectBadge.title =
      stats.detectEveryN > 1
        ? `Detections per second (running every ${stats.detectEveryN}th frame to protect fps)`
        : 'Detections per second';
  }

  setStatus(text: string): void {
    this.statusLine.textContent = text;
  }

  showError(text: string): void {
    this.errorLine.textContent = text;
    this.errorLine.hidden = false;
  }

  clearError(): void {
    this.errorLine.hidden = true;
    this.errorLine.textContent = '';
  }
}
