import fs from 'fs';
import path from 'path';

import { describe, it, expect, beforeEach, afterAll, vi } from 'vitest';

// Isolate filesystem writes to a per-process tempdir — same shape as
// `ipc-auth.test.ts`. `vi.mock` hoists above top-level consts, so the
// tempdir is computed inside `vi.hoisted` and shared with the factory.
const { TEST_DATA_DIR, TEST_STORE_DIR } = vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const osMod = require('os') as typeof import('os');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pathMod = require('path') as typeof import('path');
  const base = pathMod.join(
    osMod.tmpdir(),
    `nanoclaw-ops-fetch-test-${process.pid}`,
  );
  return {
    TEST_DATA_DIR: pathMod.join(base, 'data'),
    TEST_STORE_DIR: pathMod.join(base, 'store'),
  };
});
vi.mock('../config.js', async () => {
  const actual =
    await vi.importActual<typeof import('../config.js')>('../config.js');
  return { ...actual, DATA_DIR: TEST_DATA_DIR, STORE_DIR: TEST_STORE_DIR };
});

// Both handlers delegate the actual work to a child process (snitchmd via
// execFile, or the named-sidecar runner). These tests pin the HANDLER
// contract — registration, the isMain gate, payload validation, and the
// error-envelope shape — so the spawn boundary is stubbed.
const { mockRunSidecar } = vi.hoisted(() => ({ mockRunSidecar: vi.fn() }));
vi.mock('../sidecar-runner.js', () => ({ runSidecar: mockRunSidecar }));

import { hasIpcHandler, _resetIpcRegistryForTests } from '../ipc-registry.js';
import { registerOpsFetchIpcHandlers } from './ops-fetch.js';
import { processTaskIpc } from '../ipc.js';
import { _resetCoreIpcHandlersForTests } from './index.js';
import type { IpcDeps } from '../ipc.js';
import type { IpcTaskPayload } from '../ipc-registry.js';

const SOURCE_GROUP = 'main-group';
const REQUEST_ID = 'req-1';

function readEnvelope(): Record<string, unknown> | undefined {
  const p = path.join(
    TEST_DATA_DIR,
    'ipc',
    SOURCE_GROUP,
    'input-default',
    `_script_result_${REQUEST_ID}.json`,
  );
  if (!fs.existsSync(p)) return undefined;
  return JSON.parse(fs.readFileSync(p, 'utf-8')) as Record<string, unknown>;
}

const deps = {} as IpcDeps;

async function run(
  data: Partial<IpcTaskPayload> & { type: string },
  isMain: boolean,
): Promise<void> {
  await processTaskIpc(data as IpcTaskPayload, SOURCE_GROUP, isMain, deps);
}

beforeEach(() => {
  fs.rmSync(TEST_DATA_DIR, { recursive: true, force: true });
  fs.mkdirSync(TEST_DATA_DIR, { recursive: true });
  mockRunSidecar.mockReset();
  _resetIpcRegistryForTests();
  _resetCoreIpcHandlersForTests();
});

afterAll(() => {
  fs.rmSync(path.dirname(TEST_DATA_DIR), { recursive: true, force: true });
});

describe('registerOpsFetchIpcHandlers', () => {
  it('claims both process-delegating command names', () => {
    registerOpsFetchIpcHandlers();
    expect(hasIpcHandler('fetch_markdown')).toBe(true);
    expect(hasIpcHandler('run_sidecar')).toBe(true);
  });
});

describe('fetch_markdown handler', () => {
  it('writes an error envelope for a missing url instead of hanging the caller', async () => {
    // The in-container caller polls for the result file, so a rejected
    // payload must still produce one — silence reads as a hang.
    await run({ type: 'fetch_markdown', requestId: REQUEST_ID }, true);
    expect(String(readEnvelope()?.error)).toBeTruthy();
  });

  it('rejects a non-http scheme', async () => {
    await run(
      {
        type: 'fetch_markdown',
        requestId: REQUEST_ID,
        url: 'file:///etc/passwd',
      },
      true,
    );
    expect(String(readEnvelope()?.error)).toBeTruthy();
  });

  it('does nothing without a requestId', async () => {
    // Whole body is requestId-gated; no reply path means no work.
    await run({ type: 'fetch_markdown', url: 'https://example.com' }, true);
    expect(readEnvelope()).toBeUndefined();
  });
});

describe('run_sidecar handler', () => {
  it('refuses a non-main caller and never spawns a sidecar', async () => {
    // The sidecar runs privileged with registry-defined host bind-mounts,
    // so the host-side isMain gate is the blast-radius control.
    await run(
      { type: 'run_sidecar', requestId: REQUEST_ID, name: 'audible' },
      false,
    );
    expect(mockRunSidecar).not.toHaveBeenCalled();
    expect(readEnvelope()).toBeUndefined();
  });

  it('rejects a missing sidecar name with an actionable envelope', async () => {
    await run({ type: 'run_sidecar', requestId: REQUEST_ID }, true);
    expect(String(readEnvelope()?.error)).toContain('"name"');
    expect(mockRunSidecar).not.toHaveBeenCalled();
  });

  it('rejects a non-array flags payload rather than crashing the runner', async () => {
    await run(
      {
        type: 'run_sidecar',
        requestId: REQUEST_ID,
        name: 'audible',
        flags: '--dry-run',
      },
      true,
    );
    expect(String(readEnvelope()?.error)).toContain('"flags"');
    expect(mockRunSidecar).not.toHaveBeenCalled();
  });

  it('rejects a flags array containing a non-string', async () => {
    await run(
      {
        type: 'run_sidecar',
        requestId: REQUEST_ID,
        name: 'audible',
        flags: ['--dry-run', 42],
      },
      true,
    );
    expect(String(readEnvelope()?.error)).toContain('"flags"');
    expect(mockRunSidecar).not.toHaveBeenCalled();
  });

  it('forwards a well-formed request to the sidecar runner', async () => {
    // The handler is fire-and-forget: it chains `.then` off runSidecar's
    // promise, so the stub has to resolve one.
    mockRunSidecar.mockResolvedValue({ stdout: 'done' });
    await run(
      {
        type: 'run_sidecar',
        requestId: REQUEST_ID,
        name: 'audible',
        flags: ['--dry-run'],
      },
      true,
    );
    expect(mockRunSidecar).toHaveBeenCalledOnce();
  });
});
