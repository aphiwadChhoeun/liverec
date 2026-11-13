# Augmentation modules

This directory holds the pluggable visual effects ("augmentation modules") of
LiveRec. A module receives the composited canvas each frame, along with the
latest detection data and pixel-space helpers, and draws whatever it likes.
**The core pipeline (camera → detector → renderer) never changes when modules
are added, removed or toggled.**

## How it works

Per rendered frame, the `Renderer` (in `src/app/Renderer.ts`) does:

1. draw the video frame (mirrored for the selfie view),
2. draw the enabled wireframe layers,
3. call `update()` on every **enabled** module, **in registration order**.

A module that throws during `update()` is logged, surfaced in the UI and
disabled for the rest of the session — one broken effect can never kill the
render loop.

## The contract

```ts
export interface AugmentationModule {
  /** Stable unique id (used by the registry and the UI). */
  id: string;
  /** Human-readable name shown in the UI. */
  name: string;
  /** One-line description shown under the checkbox. */
  description?: string;
  /** Toggled at runtime — disabled modules are skipped entirely. */
  enabled: boolean;
  /** Called once per rendered frame, in registration order. */
  update(
    ctx: CanvasRenderingContext2D,
    result: DetectionResult,
    context: ModuleContext,
  ): void;
  onEnable?(): void;   // fired when toggled on (state transition only)
  onDisable?(): void;  // fired when toggled off
  dispose?(): void;    // fired when unregistered / on app teardown
}
```

> **Note on `update`'s signature:** the original project sketch passed the
> `<video>` element as the third argument. It has been folded into
> `ModuleContext` (`context.video`) so that same object can also carry the
> pixel-space helpers — one parameter, everything a module needs.

## ModuleContext

```ts
export interface ModuleContext {
  video: HTMLVideoElement;          // the frame source being composited
  mirrored: boolean;                // is the view mirrored (selfie)?
  canvasSize: { width; height };    // current canvas pixel size
  time: number;                     // seconds since the renderer started
  dt: number;                       // seconds since the previous frame
  toCanvas(point: Point3D): Point2D;        // normalized → canvas pixels
  landmarkToPixel(point: Point3D): Point2D; // alias of toCanvas
  faceOvalPath(faceIndex?: number, opts?: { scale?: number }): Path2D | null;
}
```

- **Landmarks in `result` are normalized [0,1]** relative to the *un-mirrored*
  video. Always convert through `toCanvas` / `landmarkToPixel` — they apply
  the mirroring and the current canvas size for you. Raw math like
  `lm.x * canvasSize.width` will be wrong whenever mirroring is on.
- **`faceOvalPath(0)`** returns a `Path2D` around the first face (or `null`
  when no face is tracked). `scale: 1.1` pads it 10% around its centroid —
  handy for clip regions.
- **`time` / `dt`** let you animate independently of frame rate; clamp `dt`
  (huge gaps appear after tab switches).
- **Blendshapes** (`result.face.blendshapes`, when the detector provides
  them) are per-face arrays of `{ categoryName, score }` — gate effects on
  e.g. `mouthSmileLeft` scores.
- **`result.segmentation`** is reserved for future mask-producing backends
  (person segmentation, background replacement). Check for its presence
  before using it.

## Adding a module

1. Drop a file in this directory, e.g. `src/modules/GlowEyes.ts`:

   ```ts
   import type { AugmentationModule, ModuleContext } from './AugmentationModule';
   import type { DetectionResult } from '../app/types';

   export class GlowEyes implements AugmentationModule {
     id = 'glow-eyes';
     name = 'Glowing eyes';
     description = 'Neon glow around the tracked irises.';
     enabled = false; // start unchecked; the user opts in

     update(
       ctx: CanvasRenderingContext2D,
       result: DetectionResult,
       context: ModuleContext,
     ): void {
       const face = result.face?.landmarks[0];
       if (!face) return;

       ctx.save();
       try {
         ctx.globalCompositeOperation = 'lighter';
         ctx.fillStyle = 'rgba(56, 189, 248, 0.55)';
         for (const iris of [468, 473]) { // left/right iris landmarks
           const lm = face[iris];
           if (!lm) continue;
           const p = context.toCanvas(lm); // mirror-aware pixels!
           ctx.beginPath();
           ctx.arc(p.x, p.y, 12, 0, Math.PI * 2);
           ctx.fill();
         }
       } finally {
         ctx.restore(); // never leak ctx state to the next module
       }
     }

     onDisable(): void {
       // clear caches / pause audio / …
     }
   }
   ```

2. Register it with **one line** in `src/main.ts`:

   ```ts
   registerModule(new GlowEyes());
   ```

A checkbox appears in the "Augmentation modules" panel automatically. Toggling
fires `onEnable`/`onDisable`; refresh the panel by calling
`controls.setModules(getModules())` if you register at runtime.

## Guidelines

- **Save/restore the ctx.** Modules run in sequence on one shared context;
  a leaked `filter`, `globalAlpha` or `globalCompositeOperation` corrupts
  everything drawn after you.
- **No-op cheaply** when your anchors aren't tracked (no face / no pose):
   the render loop is a hot path.
- **Avoid per-frame allocations** — pool particles/buffers, reuse objects
   (see `ParticleTrail` for a swap-remove pool).
- **Respect `prefers-reduced-motion`** if your effect moves anything
   (see `ParticleTrail`'s constructor for the pattern).
- **Errors are tolerated but fatal for the module** — a throw disables your
   module for the session, so fail gracefully where you can.
- Want to filter pose landmarks by confidence? Use
   `pointVisible(lm, 0.5)` from `src/utils/geometry.ts`.

## Registry API (`src/modules/registry.ts`)

| Function | Purpose |
| --- | --- |
| `registerModule(module)` | Add a module (throws on duplicate ids). |
| `getModules()` | Snapshot of the registry in draw order. |
| `getModule(id)` | Look up a single module. |
| `setModuleEnabled(id, on)` | Toggle + fire lifecycle hooks (rolls back on hook errors). |
| `unregisterModule(id)` | Remove, disable and dispose a module. |
| `clearModules()` | Dispose everything (app teardown / tests). |

## Reference implementations

- **`SkinSmooth.ts`** — clipping to `faceOvalPath`, mirror-aware video
  re-draw, `ctx.filter` with a feature-detect fallback. Explicitly marked as
  a *placeholder* for a real shader pipeline.
- **`ParticleTrail.ts`** — anchor-based spawning (pose wrists with hand-wrist
  fallback), pooled particles, `time`/`dt` driven motion, additive blending,
  reduced-motion awareness.
- **`IronManMask.ts`** — fits a red-and-gold helmet to the face: landmark-based
  roll/scale/yaw estimation (forehead, chin, cheekbones, eyes, nose, mouth),
  EMA-smoothed per-frame fit, and a layered canvas "material" pipeline (cast
  shadow, banded metallic gradients, embossed panel seams, yaw-swept
  speculars, glowing angular eye slits, recessed mouth grill). Appear/ghost
  fades mask detection drop-outs, and per-face `Path2D`/gradient caching keeps
  steady-state rendering allocation-free. Exports `estimateIronManFit` as a
  DOM-free helper covered by unit tests.
- **`PandaFace.ts`** — fits an opaque white panda face to the tracked face oval
  with even-odd cut-outs for the eyes, plus black ears, slanted eye patches, a
  glossy button nose and "w" smile. EMA-smoothed fit, appear/fade ramp and
  per-face path/gradient caching. Exports `estimatePandaFit` as a DOM-free
  helper covered by unit tests.

Future extension points already supported by the data contracts (no core
changes needed): mesh warping (478 face points + `z`), background
replacement via `result.segmentation`, landmark-attached 3D/sprite objects,
blendshape-gated makeup, MediaRecorder canvas export.
