import fs from 'fs';

import { describe, it, expect, afterAll, beforeEach, vi } from 'vitest';

// Isolate filesystem writes (the dispatcher's main-only error envelope
// goes through `scriptResultPath`, which builds paths under `DATA_DIR`)
// to a per-process tempdir — same pattern as ipc-auth.test.ts.
const { TEST_DATA_DIR } = vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const osMod = require('os') as typeof import('os');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pathMod = require('path') as typeof import('path');
  return {
    TEST_DATA_DIR: pathMod.join(
      osMod.tmpdir(),
      `nanoclaw-ipc-registry-test-${process.pid}`,
    ),
  };
});
vi.mock('./config.js', async () => {
  const actual =
    await vi.importActual<typeof import('./config.js')>('./config.js');
  return {
    ...actual,
    DATA_DIR: TEST_DATA_DIR,
  };
});

import path from 'path';

import { registerCoreIpcHandlers } from './ipc-handlers/index.js';
import {
  dispatchIpcTask,
  hasIpcHandler,
  registerIpcHandler,
  type IpcHandlerContext,
} from './ipc-registry.js';
import type { IpcDeps } from './ipc.js';

afterAll(() => {
  fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
});

// The registry is module-global state shared across this file's tests, so
// every registered name here is unique per test. Only the dispatch
// contract is under test — a minimal deps stub is enough because the
// dispatcher itself never touches deps (handlers do).
const deps = {} as IpcDeps;

function ctx(overrides: Partial<IpcHandlerContext>): IpcHandlerContext {
  return {
    data: { type: 'unregistered_command' },
    sourceGroup: 'some-group',
    isMain: false,
    deps,
    ...overrides,
  };
}

beforeEach(() => {
  fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
});

describe('registerIpcHandler / dispatchIpcTask', () => {
  it('returns false for a type with no registered handler', async () => {
    const handled = await dispatchIpcTask(
      ctx({ data: { type: 'unregistered_command' } }),
    );
    expect(handled).toBe(false);
  });

  it('invokes the registered handler with the full context', async () => {
    const seen: IpcHandlerContext[] = [];
    registerIpcHandler('test_echo', {
      handler: (c) => {
        seen.push(c);
      },
    });
    const context = ctx({
      data: { type: 'test_echo', message: 'hi' },
      sourceGroup: 'group-a',
      isMain: true,
    });
    const handled = await dispatchIpcTask(context);
    expect(handled).toBe(true);
    expect(seen).toHaveLength(1);
    expect(seen[0].data.message).toBe('hi');
    expect(seen[0].sourceGroup).toBe('group-a');
    expect(seen[0].isMain).toBe(true);
    expect(seen[0].deps).toBe(deps);
  });

  it('throws on duplicate registration of the same command name', () => {
    registerIpcHandler('test_dup', { handler: () => {} });
    expect(() => registerIpcHandler('test_dup', { handler: () => {} })).toThrow(
      /already registered: test_dup/,
    );
  });

  it('propagates a rejection from the handler to the dispatcher caller', async () => {
    registerIpcHandler('test_throws', {
      handler: () => {
        throw new SyntaxError('bad payload');
      },
    });
    await expect(
      dispatchIpcTask(ctx({ data: { type: 'test_throws' } })),
    ).rejects.toThrow('bad payload');
  });
});

describe('requiresMain gate', () => {
  it('blocks a non-main caller before the handler runs', async () => {
    const handler = vi.fn();
    registerIpcHandler('test_admin_only', { requiresMain: true, handler });
    const handled = await dispatchIpcTask(
      ctx({ data: { type: 'test_admin_only' }, isMain: false }),
    );
    expect(handled).toBe(true);
    expect(handler).not.toHaveBeenCalled();
  });

  it('writes an error envelope for a blocked request that carries a requestId', async () => {
    registerIpcHandler('test_admin_envelope', {
      requiresMain: true,
      handler: vi.fn(),
    });
    await dispatchIpcTask(
      ctx({
        data: { type: 'test_admin_envelope', requestId: 'req-1' },
        sourceGroup: 'group-b',
        isMain: false,
      }),
    );
    const resultPath = path.join(
      TEST_DATA_DIR,
      'ipc',
      'group-b',
      'input-default',
      '_script_result_req-1.json',
    );
    const envelope = JSON.parse(fs.readFileSync(resultPath, 'utf-8'));
    expect(envelope.error).toMatch(/test_admin_envelope is admin-tile only/);
  });

  it('does not write any envelope for a blocked fire-and-forget request', async () => {
    registerIpcHandler('test_admin_silent', {
      requiresMain: true,
      handler: vi.fn(),
    });
    await dispatchIpcTask(
      ctx({
        data: { type: 'test_admin_silent' },
        sourceGroup: 'group-c',
        isMain: false,
      }),
    );
    expect(fs.existsSync(path.join(TEST_DATA_DIR, 'ipc', 'group-c'))).toBe(
      false,
    );
  });

  it('lets a main caller through to the handler', async () => {
    const handler = vi.fn();
    registerIpcHandler('test_admin_main', { requiresMain: true, handler });
    const handled = await dispatchIpcTask(
      ctx({ data: { type: 'test_admin_main' }, isMain: true }),
    );
    expect(handled).toBe(true);
    expect(handler).toHaveBeenCalledOnce();
  });
});

describe('registerCoreIpcHandlers', () => {
  it('registers the task-lifecycle slice and is idempotent', () => {
    registerCoreIpcHandlers();
    // A second call must not throw duplicate-registration errors.
    registerCoreIpcHandlers();
    for (const name of [
      'schedule_task',
      'pause_task',
      'resume_task',
      'cancel_task',
      'update_task',
    ]) {
      expect(hasIpcHandler(name)).toBe(true);
    }
  });
});
