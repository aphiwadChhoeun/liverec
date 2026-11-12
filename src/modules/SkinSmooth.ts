/**
 * Demo augmentation module: softens skin inside the tracked face oval.
 *
 * PLACEHOLDER PIPELINE — this redraws the video frame with a canvas `blur()`
 * filter clipped to the (slightly padded) face oval. It is intentionally
 * simple: replace the `update()` body with a real bilateral / separable blur
 * or a WebGL shader pass for production quality. The landmark plumbing
 * (oval path from `ModuleContext`, clip, mirror-aware redraw) is the part to
 * keep. In browsers without `ctx.filter` support it falls back to a cheap
 * semi-transparent double-exposure.
 */
import type { AugmentationModule, ModuleContext } from './AugmentationModule';
import type { DetectionResult } from '../app/types';

export class SkinSmooth implements AugmentationModule {
  id = 'skin-smooth';
  name = 'Skin smoothing';
  description = 'Softens the tracked face oval (placeholder blur pipeline).';
  enabled = false;

  private filterChecked = false;
  private filterSupported = false;

  update(
    ctx: CanvasRenderingContext2D,
    result: DetectionResult,
    context: ModuleContext,
  ): void {
    const face = result.face?.landmarks[0];
    if (!face || face.length === 0) return;

    const path = context.faceOvalPath(0, { scale: 1.08 });
    if (!path) return;

    if (!this.filterChecked) {
      // `ctx.filter` is unsupported in older Safari; detect once and degrade.
      this.filterChecked = true;
      this.filterSupported = typeof ctx.filter === 'string';
      if (!this.filterSupported) {
        console.warn(
          '[SkinSmooth] ctx.filter unsupported — using alpha blend fallback.',
        );
      }
    }

    const { width, height } = context.canvasSize;

    ctx.save();
    try {
      ctx.clip(path);
      if (this.filterSupported) {
        ctx.filter = 'blur(5px) brightness(1.04) saturate(1.04)';
      } else {
        ctx.globalAlpha = 0.35;
      }
      if (context.mirrored) {
        ctx.translate(width, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(context.video, 0, 0, width, height);
    } finally {
      ctx.restore(); // also resets ctx.filter / alpha / transform
    }
  }

  onDisable(): void {
    // Nothing stateful to clear — the canvas redraws every frame.
  }
}
