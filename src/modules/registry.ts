/**
 * Module registry. Add an effect by dropping its file into `src/modules/`
 * and registering one line in `src/main.ts`:
 *
 *   registerModule(new MyEffect());
 *
 * The renderer iterates the registry every frame, so registration order is
 * the draw order.
 */
import type { AugmentationModule } from './AugmentationModule';

const modules: AugmentationModule[] = [];

/** Register a module. Throws on duplicate ids. */
export function registerModule(module: AugmentationModule): void {
  if (modules.some((m) => m.id === module.id)) {
    throw new Error(`Augmentation module "${module.id}" is already registered.`);
  }
  modules.push(module);
}

/**
 * All registered modules in draw order. Returns a defensive snapshot, so
 * callers cannot mutate the registry by accident (the renderer re-queries
 * every frame and picks up runtime registrations automatically).
 */
export function getModules(): readonly AugmentationModule[] {
  return modules.slice();
}

export function getModule(id: string): AugmentationModule | undefined {
  return modules.find((m) => m.id === id);
}

/**
 * Toggle a module, firing its lifecycle hooks on state transitions only.
 * Returns `false` when the id is unknown or a lifecycle hook threw (in which
 * case the flag is rolled back).
 */
export function setModuleEnabled(id: string, enabled: boolean): boolean {
  const module = getModule(id);
  if (!module) return false;
  if (module.enabled === enabled) return true;

  module.enabled = enabled;
  try {
    if (enabled) {
      module.onEnable?.();
    } else {
      module.onDisable?.();
    }
  } catch (error) {
    console.error(`[registry] lifecycle hook of module "${id}" threw:`, error);
    module.enabled = !enabled;
    return false;
  }
  return true;
}

/** Remove (and dispose) a module. Returns `false` for unknown ids. */
export function unregisterModule(id: string): boolean {
  const index = modules.findIndex((m) => m.id === id);
  if (index === -1) return false;

  const [module] = modules.splice(index, 1);
  if (!module) return false;

  if (module.enabled) {
    try {
      module.onDisable?.();
    } catch (error) {
      console.error(`[registry] onDisable of module "${id}" threw:`, error);
    }
  }
  try {
    module.dispose?.();
  } catch (error) {
    console.error(`[registry] dispose of module "${id}" threw:`, error);
  }
  return true;
}

/** Disposes and clears everything (app teardown + tests). */
export function clearModules(): void {
  while (modules.length > 0) {
    const module = modules[0];
    if (!module) break;
    unregisterModule(module.id);
  }
}
