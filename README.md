# LiveRec · Webcam Face & Pose Studio

Real-time webcam face & pose recognition in the browser, with a wireframe
overlay (face mesh · pose skeleton · hand landmarks) and an **extensible
augmentation-module pipeline** — new visual effects plug in without touching
the core detection code.

Everything runs client-side. The MediaPipe WASM runtime and the `.task` models
are fetched from CDN on first load; video never leaves the device.

## Quick start

```bash
npm install
npm run dev        # http://localhost:5173
```

> The browser requires a **secure context** for camera access — `localhost`
> counts. When hosting remotely, serve over HTTPS.

```bash
npm run build      # static bundle in dist/ (typechecks first)
npm run preview    # serve the built bundle
npm test           # vitest unit tests
```

## What you get

- Mirrored selfie view of the webcam at up to 1280×720.
- Face mesh (478-point tessellation + contours), pose skeleton (33 landmarks)
  and hand landmarks (2 × 21), all toggleable live.
- FPS + detection-rate badges; adaptive detection throttling keeps the preview
  smooth on slower machines.
- Camera switching mid-stream, mirror toggle, PNG snapshots.
- A registry of augmentation modules: **Skin smoothing** (placeholder blur
  pipeline), **Particle trail** (wrist-anchored particles) and **Iron Man mask**
  (a red-and-gold helmet tracked onto your face, with metallic shading,
  head-yaw-aware lighting and glowing eyes), all toggleable live — and all
  examples of how to add your own (see
  [`src/modules/README.md`](src/modules/README.md)).

## Architecture

```
src/
  main.ts                 # bootstrap + orchestration (start/stop, switching, errors)
  app/
    types.ts              # shared contracts: DetectionResult, Point3D, LayerId, …
    Camera.ts             # getUserMedia, device enumeration, mid-stream switching
    Detector.ts           # Detector interface + MediaPipeDetector (GPU→CPU fallback)
    Renderer.ts           # canvas + rAF loop: video → wireframe → modules
  modules/                # augmentation plugins (see modules/README.md)
    AugmentationModule.ts # the module contract + ModuleContext helpers
    registry.ts           # register/unregister/enable + lifecycle
    SkinSmooth.ts         # demo: face-oval blur (placeholder pipeline)
    ParticleTrail.ts      # demo: wrist particle trail
    IronManMask.ts        # demo: landmark-tracked Iron Man helmet
  ui/
    Controls.ts           # buttons, dropdowns, checkboxes, badges, status/error
  utils/
    geometry.ts           # pixel conversion, face-oval tracing, bounds, lerp
tests/                    # vitest: geometry + registry
```

Data flow per frame:

```
<video> ──► Detector.detect() ──► DetectionResult (normalized [0,1] coords)
                                        │
              Renderer (rAF loop) ──────┤  1. draw video (mirrored)
                                        │  2. draw wireframe layers
                                        │  3. run enabled modules (in order)
                                        ▼
                                   <canvas> overlay
```

Key design decisions:

- **Landmarks stay normalized** until the renderer converts them (mirroring is
  applied there, once) — modules get pixel-space helpers via `ModuleContext`
  (`toCanvas`, `faceOvalPath`, …) instead of re-doing the math.
- **Detection is decoupled from rendering.** The renderer draws every rAF
  frame but only runs detection every Nth frame (N adapts 1–4 based on fps),
  and the detector skips a frame entirely while a previous pass is in flight.
- **The `Detector` interface is swappable** — a WebSocket or TF.js backend can
  replace MediaPipe without touching the renderer or modules.
- **Graceful degradation everywhere:** GPU delegate → CPU fallback per
  landmarker; a failed model disables only its feature (the layer checkbox
  greys out with the reason); a failed *detector* still leaves the plain
  camera preview; a throwing module is auto-disabled instead of killing the
  loop.

## Adding an augmentation module

1. Create `src/modules/MyEffect.ts` implementing `AugmentationModule`
   (template: [`src/modules/README.md`](src/modules/README.md)).
2. Register it in `src/main.ts`:

   ```ts
   registerModule(new MyEffect());
   ```

That's it — a checkbox appears in the UI, and the renderer calls your
`update()` every frame in registration order. No core files change.

## Swapping the detector backend

Implement the three-method `Detector` interface (`init` / `detect` / `dispose`)
in `src/app/Detector.ts` (or a new file next to it) and pass it to the
`Renderer` in `src/main.ts`. `DetectionResult` is the only contract — it
already carries arrays for multi-person scenes, blendshapes and an opaque
`segmentation` slot for future mask-producing backends.

## Performance notes

- Detection and rendering are separate cadences; the video preview never
  blocks on the model.
- The wireframe is drawn as batched `beginPath → moveTo/lineTo → stroke`
  calls (no per-connection draw calls, no per-frame Path2D allocations).
- `ModuleContext` is a reused object; particles use a pooled array with
  swap-removal; `getModules()` snapshots are tiny.
- If fps drops below ~22, detection automatically runs every 2nd–4th frame
  until it recovers (the `detect` badge shows the current cadence).

## Accessibility

- All controls are native, keyboard-operable elements with visible focus.
- Status changes announce via `aria-live="polite"`; errors via `role="alert"`.
- `prefers-reduced-motion` disables the particle module and all UI
  transitions.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| "Camera permission denied" | Allow camera access for the site, then press Start again. |
| "Camera API unavailable" | Page is not in a secure context — use `localhost` or HTTPS. |
| "Detection models failed to load …" | No internet on first run (models/WASM come from CDN). The preview keeps working without overlays; retry by reloading. |
| Models load but layers are greyed out | That specific model failed — hover the checkbox for the reason; the rest keep working. |
| `detect …/s` badge shows "every Nth frame" | Adaptive throttling kicked in to protect fps. |
| Skin smoothing looks like a hard-edged ring | It's a placeholder clip+blur demo — see `src/modules/SkinSmooth.ts`. |

## Tests

```bash
npm test
```

Covers the geometry helpers (pixel conversion, mirroring, the canonical face
oval — cross-checked against the MediaPipe package data — bounds, centroids,
lerp), the Iron Man helmet fit (landmark anchoring, roll, head yaw, mirroring,
fallbacks and degenerate faces) and the module registry (ordering, duplicates,
lifecycle transitions, error rollback, disposal).

## Acceptance criteria mapping

- **Mirrored selfie view** — renderer flips the video + all landmark drawing;
  a mirror checkbox reflects/overrides the camera's facing-mode hint.
- **Wireframe ≥ 24 fps on mid-range hardware** — GPU-delegate MediaPipe with
  adaptive detection throttling; fps badge turns amber below 24.
- **Live toggling** — layers and modules are read every frame; no restart.
- **Mid-stream camera switching** — dropdown re-requests `getUserMedia` with
  the new device; the canvas auto-resizes to the new resolution.
- **New module = `modules/` file + one registration line** — enforced by the
  registry design (see the modules README).
- **No console errors / errors surfaced in UI** — happy path is silent;
  failures land in the `role="alert"` error line, never in an uncaught throw.
