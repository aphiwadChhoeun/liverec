# Prompt: Webcam Face & Pose Recognition with Extensible Augmentation

You are an expert full-stack frontend engineer. Implement a browser-based application that
captures webcam video, detects faces and body pose in real time, renders a wireframe
skeleton / face mesh overlay on a canvas, and is architected so that additional
"augmentation modules" (skin smoothing, virtual makeup, particle effects, AR filters,
etc.) can be plugged in without touching the core detection pipeline.

## 1. Goals

- Real-time (≥ 24 fps on a mid-range laptop) face + pose detection from the webcam.
- Render a live wireframe (face mesh + body skeleton + hand landmarks) over the video.
- Expose a clean, module-based API so new visual effects can consume the detected
  landmarks/mesh and draw onto the same canvas.
- Keep everything client-side (no server required); models loaded from CDN or local assets.
- Responsive UI: start/stop, camera selection, FPS counter, toggle layers, screenshot.

## 2. Recommended Tech Stack

- **Framework:** Vanilla TypeScript + Vite (or React + Vite if the repo already uses React).
  Keep DOM manipulation thin; the heavy work lives in a render loop.
- **Detection:** MediaPipe via `@mediapipe/tasks-vision` (Face Landmarker + Pose Landmarker
  + Hand Landmarker). These run on WASM + GPU delegate and give 3D landmarks + blendshapes.
  - Alternative: TensorFlow.js + `@tensorflow-models/face-landmarks-detection` and
    `@tensorflow-models/pose-detection` (MoveNet). Use MediaPipe first; keep a swappable
    `Detector` interface so the backend can be swapped.
- **Rendering:** A single `<canvas>` overlaid on a `<video>` element, drawn with 2D context
  (Canvas 2D). Use `requestAnimationFrame` loop. Optional WebGL layer later for heavy FX.
- **Styling:** Plain CSS or Tailwind; dark theme default.

## 3. Architecture (must follow)

```
src/
  main.ts                 # bootstrap
  app/
    Camera.ts             # getUserMedia, device enumeration, stream lifecycle
    Renderer.ts           # canvas setup, rAF loop, layer compositing
    Detector.ts           # interface + MediaPipe implementation
    types.ts              # shared types (Landmark, DetectionResult, ...)
  modules/                # augmentation plugins (empty scaffolding + 1 demo)
    AugmentationModule.ts # interface
    SkinSmooth.ts         # example: face-region skin smoothing (canvas filter)
    ParticleTrail.ts      # example: particles following wrist landmarks
  ui/
    Controls.ts           # buttons, toggles, FPS
  utils/
    geometry.ts           # landmark helpers, triangulation, bounding boxes
```

### Core interfaces (define these first)

```ts
// types.ts
export interface Point3D { x: number; y: number; z: number; visibility?: number; }
export interface DetectionResult {
  timestamp: number;
  face?: { landmarks: Point3D[][]; blendshapes?: any; };
  pose?: { landmarks: Point3D[]; worldLandmarks?: Point3D[]; };
  hands?: { landmarks: Point3D[][]; handedness: string[]; };
  videoSize: { width: number; height: number; };
}

// Detector.ts
export interface Detector {
  init(): Promise<void>;
  detect(video: HTMLVideoElement, now: number): Promise<DetectionResult | null>;
  dispose(): void;
}

// AugmentationModule.ts
export interface AugmentationModule {
  id: string;
  name: string;
  enabled: boolean;
  // Called once per frame with the latest detection + canvas context.
  update(ctx: CanvasRenderingContext2D, result: DetectionResult, video: HTMLVideoElement): void;
  // Optional lifecycle
  onEnable?(): void;
  onDisable?(): void;
  dispose?(): void;
}
```

## 4. Implementation Steps

1. **Scaffold** Vite + TS project; add `@mediapipe/tasks-vision`. Create the file tree above.
2. **Camera module**: enumerate `navigator.mediaDevices.enumerateDevices()`, request
   `getUserMedia({ video: { facingMode: 'user', width: 1280, height: 720 } })`, attach to
   `<video autoplay playsinline muted>`. Handle permission errors and device switching.
3. **Detector (MediaPipe)**:
   - Load `FaceLandmarker`, `PoseLandmarker`, `HandLandmarker` from the vision bundle
     (use `FilesetResolver.forVisionTasks` + CDN wasm).
   - Run in `VIDEO` running mode; call `detectForVideo(video, timestamp)`.
   - Normalize outputs into `DetectionResult`. Convert normalized [0,1] coords to pixels
     using video dimensions inside the Renderer (keep detector output normalized).
   - Throttle: skip frames if previous detection not finished (use a busy flag).
4. **Renderer loop**:
   - Each frame: draw video (mirrored horizontally for selfie view) → draw wireframe →
     run each enabled augmentation module in order → update FPS.
   - Wireframe drawing helpers: face mesh connections (MediaPipe FACEMESH_TESSELATION /
     FACEMESH_CONTOURS), pose connections (POSE_CONNECTIONS), hand connections.
   - Mirror handling: when drawing landmarks, flip X if mirrored.
5. **UI controls**: Start/Stop camera, camera dropdown, checkboxes per layer
   (face mesh, pose skeleton, hands), checkboxes per augmentation module, FPS display,
   "Save snapshot" (canvas.toDataURL).
6. **Augmentation scaffolding**:
   - `SkinSmooth`: demonstrate sampling the face oval region and applying a canvas
     filter (e.g. `ctx.filter = 'blur(2px)'` clipped to face path) — clearly marked as
     a placeholder to be replaced with a real shader/pipeline.
   - `ParticleTrail`: particles spawned at wrist landmarks, fading over time.
   - Provide a registry so modules can be added by dropping a file in `modules/` and
     registering in `main.ts`.
7. **Performance**: cap detection to every Nth frame if FPS drops; reuse buffers; avoid
   allocating arrays per frame where possible; expose `performance.now()` timing.

## 5. Wireframe / Augmentation Extension Points (open for expansion)

Design so these future features plug in with NO core changes:

- **Skin augmentation:** modules receive the face landmarks + blendshapes; can build a
  face oval mask, apply color grading, virtual makeup, or texture overlays.
- **Special effects:** particle systems, glow, distortion (mesh warping), background
  segmentation (person mask → replace background), 3D objects attached to landmarks.
- **Export:** record canvas to MediaRecorder for video export.
- **Multi-person:** detector interface already returns arrays; renderer loops over them.
- **Custom models:** swap `Detector` implementation (e.g. a remote/WebSocket detector).

To support this, the `DetectionResult` must include enough data (landmarks, blendshapes,
segmentation mask if available) and the Renderer must expose the canvas context + video
frame to modules. Add a `ModuleContext` with helpers: `toCanvas(point)`, `faceOvalPath()`,
`landmarkToPixel()`.

## 6. Quality Requirements

- TypeScript strict mode, no `any` in public interfaces (internal casts OK with comments).
- Graceful degradation: if GPU delegate fails, fall back to CPU; if a model fails to load,
  show a clear error and disable that feature.
- Accessibility: keyboard-operable controls, ARIA labels, respect `prefers-reduced-motion`
  (disable particle effects).
- Tests: at least unit tests for geometry helpers and the module registry (Vitest).
- README with setup, run, build, and "how to add an augmentation module" guide.

## 7. Deliverables

- Working `npm run dev` that opens the app and shows live wireframe on webcam.
- `npm run build` producing a static bundle.
- A `modules/README.md` documenting the `AugmentationModule` contract and a template.

## 8. Non-Goals (do not implement now)

- Server-side processing, user accounts, cloud storage.
- Training custom models.
- Mobile-native apps (web only, but must be responsive).

## Acceptance Criteria

- [ ] Webcam feed displays with mirrored selfie view.
- [ ] Face mesh + pose skeleton + hands render in real time at ≥ 24 fps.
- [ ] Toggling layers/modules updates rendering without restart.
- [ ] Camera can be switched mid-stream.
- [ ] Adding a new module requires editing only `modules/` + one registration line.
- [ ] No console errors in the happy path; errors surfaced in UI.

When implementing, start by scaffolding the project and the interfaces, then the camera +
detector, then renderer, then UI, then modules. Commit logical steps. Ask before adding
dependencies beyond those listed.
```

Write the project into the current working directory. If a framework is already present,
adapt to it rather than replacing it.<|end|><|start|><|start|>assistant<|channel|>final<|message|>I've written a comprehensive prompt file to `face-pose-recognition-prompt.md` in your working directory. It covers:

- **Goals & tech stack** (Vite + TS, MediaPipe tasks-vision, Canvas 2D, swappable Detector interface)
- **Architecture & file tree** with core interfaces (`Detector`, `AugmentationModule`, `DetectionResult`, `ModuleContext`)
- **Step-by-step implementation order** (scaffold → camera → detector → renderer → UI → modules)
- **Extension points** explicitly designed for skin augmentation, special effects, multi-person, custom models, and video export
- **Quality requirements** (strict TS, graceful degradation, accessibility, Vitest tests, README)
- **Deliverables & acceptance criteria** (checkboxes)
- **Non-goals** to keep scope bounded

Key design decisions that keep it open for expansion:
- Landmarks stay normalized in `DetectionResult`; pixel conversion happens in the Renderer via `ModuleContext` helpers like `toCanvas()` and `faceOvalPath()`.
- Modules receive the canvas context + result + video each frame, so new effects (skin smoothing, particles, mesh warping, background segmentation) plug in without core changes.
- A module registry means adding an effect is just dropping a file in `modules/` plus one registration line.

Want me to also generate a starter scaffold (Vite + TS project with the interfaces wired up), or is the prompt itself sufficient? I can also produce a variant tailored to a specific stack (React, Next.js, etc.) or add a section for WebGPU/WebGL-based FX. Just let me know. 🔧📝<|end|><|start|>tool_calls_section_begin|>|tool_calls_section_end|>|assistant to=bash<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">
ls -la "C:\Users\aphiw\personal\liverec"