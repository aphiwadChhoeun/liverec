import { beforeEach, describe, expect, it, vi } from 'vitest';
import type { AugmentationModule } from '../src/modules/AugmentationModule';
import {
  clearModules,
  getModule,
  getModules,
  registerModule,
  setModuleEnabled,
  unregisterModule,
} from '../src/modules/registry';

function makeModule(overrides: Partial<AugmentationModule> = {}): AugmentationModule {
  return {
    id: 'test-module',
    name: 'Test module',
    enabled: false,
    update: vi.fn(),
    ...overrides,
  };
}

beforeEach(() => {
  clearModules();
});

describe('registerModule', () => {
  it('adds modules in registration order', () => {
    registerModule(makeModule({ id: 'a' }));
    registerModule(makeModule({ id: 'b' }));

    expect(getModules().map((m) => m.id)).toEqual(['a', 'b']);
  });

  it('returns a snapshot that cannot mutate the registry', () => {
    registerModule(makeModule());
    const snapshot = getModules();
    (snapshot as AugmentationModule[]).push(makeModule({ id: 'evil' }));
    expect(getModules().map((m) => m.id)).toEqual(['test-module']);
  });

  it('throws on duplicate ids', () => {
    registerModule(makeModule({ id: 'dup' }));
    expect(() => registerModule(makeModule({ id: 'dup' }))).toThrowError(
      /already registered/,
    );
    expect(getModules()).toHaveLength(1);
  });
});

describe('getModule', () => {
  it('finds by id and returns undefined otherwise', () => {
    registerModule(makeModule());
    expect(getModule('test-module')?.name).toBe('Test module');
    expect(getModule('nope')).toBeUndefined();
  });
});

describe('setModuleEnabled', () => {
  it('fires lifecycle hooks only on state transitions', () => {
    const onEnable = vi.fn();
    const onDisable = vi.fn();
    registerModule(makeModule({ onEnable, onDisable }));

    expect(setModuleEnabled('test-module', true)).toBe(true);
    expect(onEnable).toHaveBeenCalledTimes(1);
    expect(getModule('test-module')?.enabled).toBe(true);

    // Same-state toggle is a no-op for hooks.
    expect(setModuleEnabled('test-module', true)).toBe(true);
    expect(onEnable).toHaveBeenCalledTimes(1);

    expect(setModuleEnabled('test-module', false)).toBe(true);
    expect(onDisable).toHaveBeenCalledTimes(1);
    expect(getModule('test-module')?.enabled).toBe(false);
  });

  it('returns false for unknown ids', () => {
    expect(setModuleEnabled('ghost', true)).toBe(false);
  });

  it('rolls the flag back when a lifecycle hook throws', () => {
    const onEnable = vi.fn(() => {
      throw new Error('boom');
    });
    registerModule(makeModule({ onEnable }));

    expect(setModuleEnabled('test-module', true)).toBe(false);
    expect(getModule('test-module')?.enabled).toBe(false);
    expect(onEnable).toHaveBeenCalledTimes(1);
  });
});

describe('unregisterModule', () => {
  it('disposes and removes the module', () => {
    const dispose = vi.fn();
    const onDisable = vi.fn();
    registerModule(makeModule({ enabled: true, onDisable, dispose }));

    expect(unregisterModule('test-module')).toBe(true);
    expect(onDisable).toHaveBeenCalledTimes(1);
    expect(dispose).toHaveBeenCalledTimes(1);
    expect(getModules()).toHaveLength(0);
    expect(unregisterModule('test-module')).toBe(false);
  });

  it('does not call onDisable for modules that were already disabled', () => {
    const onDisable = vi.fn();
    const dispose = vi.fn();
    registerModule(makeModule({ enabled: false, onDisable, dispose }));

    expect(unregisterModule('test-module')).toBe(true);
    expect(onDisable).not.toHaveBeenCalled();
    expect(dispose).toHaveBeenCalledTimes(1);
  });
});

describe('clearModules', () => {
  it('disposes everything', () => {
    const disposeA = vi.fn();
    const disposeB = vi.fn();
    registerModule(makeModule({ id: 'a', dispose: disposeA }));
    registerModule(makeModule({ id: 'b', dispose: disposeB }));

    clearModules();
    expect(getModules()).toHaveLength(0);
    expect(disposeA).toHaveBeenCalledTimes(1);
    expect(disposeB).toHaveBeenCalledTimes(1);
  });
});
