/**
 * Demo augmentation module: glowing particles that spawn at the tracked
 * wrists and fade out. Respects `prefers-reduced-motion` by staying disabled.
 *
 * Uses a pooled array with swap-removal — no per-frame allocations beyond the
 * spawned particles themselves.
 */
import type { AugmentationModule, ModuleContext } from './AugmentationModule';
import type { DetectionResult, Point2D } from '../app/types';
import {
  HAND_WRIST_INDEX,
  POSE_WRIST_INDICES,
  pointVisible,
} from '../utils/geometry';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  /** Seconds alive. */
  life: number;
  /** Time to live in seconds. */
  ttl: number;
  size: number;
  /** HSL hue (0–360). */
  hue: number;
}

const MAX_PARTICLES = 320;
const SPAWN_PER_ANCHOR = 3;
/** Slight upward drift so trails hang in the air like sparks. */
const DRIFT_PX_PER_S = -24;
const GRAVITY_PX_PER_S2 = 30;

export class ParticleTrail implements AugmentationModule {
  id = 'particle-trail';
  name = 'Particle trail';
  description = 'Glowing particles that trail from your wrists.';
  enabled: boolean;

  private particles: Particle[] = [];
  private reducedMotion = false;

  constructor(options: { enabled?: boolean } = {}) {
    this.enabled = options.enabled ?? true;
    if (typeof window !== 'undefined' && typeof window.matchMedia === 'function') {
      // Accessibility: skip motion effects entirely when the user asks for it.
      this.reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (this.reducedMotion) {
        this.enabled = false;
        this.description = 'Disabled — your system prefers reduced motion.';
      }
    }
  }

  update(
    ctx: CanvasRenderingContext2D,
    result: DetectionResult,
    context: ModuleContext,
  ): void {
    if (this.reducedMotion) return; // hard stop even if enabled programmatically

    // Clamp dt: huge gaps (tab switch) would teleport particles.
    const dt = Math.min(context.dt, 0.1);

    this.spawn(result, context);
    this.advance(dt);
    this.draw(ctx);
  }

  onEnable(): void {
    this.particles.length = 0;
  }

  dispose(): void {
    this.particles.length = 0;
  }

  /** Anchor points: pose wrists, falling back to hand wrists without a pose. */
  private collectAnchors(
    result: DetectionResult,
    context: ModuleContext,
  ): Point2D[] {
    const anchors: Point2D[] = [];

    for (const pose of result.pose?.landmarks ?? []) {
      for (const index of POSE_WRIST_INDICES) {
        const lm = pose[index];
        if (lm && pointVisible(lm)) anchors.push(context.toCanvas(lm));
      }
    }
    if (anchors.length > 0) return anchors;

    for (const hand of result.hands?.landmarks ?? []) {
      const lm = hand[HAND_WRIST_INDEX];
      if (lm) anchors.push(context.toCanvas(lm));
    }
    return anchors;
  }

  private spawn(result: DetectionResult, context: ModuleContext): void {
    if (this.particles.length >= MAX_PARTICLES) return;
    const anchors = this.collectAnchors(result, context);

    for (const anchor of anchors) {
      for (let i = 0; i < SPAWN_PER_ANCHOR; i++) {
        if (this.particles.length >= MAX_PARTICLES) return;
        this.particles.push({
          x: anchor.x,
          y: anchor.y,
          vx: (Math.random() - 0.5) * 60,
          vy: (Math.random() - 0.5) * 60 + DRIFT_PX_PER_S,
          life: 0,
          ttl: 0.6 + Math.random() * 0.8,
          size: 2 + Math.random() * 3,
          hue: (context.time * 40 + i * 17) % 360,
        });
      }
    }
  }

  private advance(dt: number): void {
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      if (!p) continue;
      p.life += dt;
      if (p.life >= p.ttl) {
        // Swap-remove: O(1) teardown, order does not matter for particles.
        const last = this.particles.pop();
        if (last && i < this.particles.length) this.particles[i] = last;
        continue;
      }
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.vy += GRAVITY_PX_PER_S2 * dt;
    }
  }

  private draw(ctx: CanvasRenderingContext2D): void {
    if (this.particles.length === 0) return;

    ctx.save();
    try {
      // Additive blending makes overlapping particles glow.
      ctx.globalCompositeOperation = 'lighter';
      for (const p of this.particles) {
        const t = p.life / p.ttl; // 0 → 1
        const alpha = (1 - t) * 0.85;
        const radius = Math.max(0.4, p.size * (1 - t * 0.6));
        ctx.fillStyle = `hsla(${p.hue.toFixed(0)}, 90%, 62%, ${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }
    } finally {
      ctx.restore();
    }
  }
}
