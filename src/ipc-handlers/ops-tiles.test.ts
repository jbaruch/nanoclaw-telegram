import fs from 'fs';
import path from 'path';

import { describe, it, expect, beforeEach, afterAll, vi } from 'vitest';

// Isolate filesystem writes to a per-process tempdir — same shape as
// `ipc-auth.test.ts`. `vi.mock` hoists above top-level consts, so the
// tempdir is computed inside `vi.hoisted` and shared with the factory.
const { TEST_DATA_DIR, TEST_GROUPS_DIR } = vi.hoisted(() => {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const osMod = require('os') as typeof import('os');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const pathMod = require('path') as typeof import('path');
  const base = pathMod.join(
    osMod.tmpdir(),
    `nanoclaw-ops-tiles-test-${process.pid}`,
  );
  return {
    TEST_DATA_DIR: pathMod.join(base, 'data'),
    TEST_GROUPS_DIR: pathMod.join(base, 'groups'),
  };
});
vi.mock('../config.js', async () => {
  const actual =
    await vi.importActual<typeof import('../config.js')>('../config.js');
  return { ...actual, DATA_DIR: TEST_DATA_DIR, GROUPS_DIR: TEST_GROUPS_DIR };
});

// The tile registry is a real on-disk surface the test environment has no
// copy of. These tests pin the HANDLER contract — registration, the isMain
// gate, the tile allowlist, and the result-envelope shape — so the registry
// read is stubbed.
const { mockGetInstalledTiles } = vi.hoisted(() => ({
  mockGetInstalledTiles: vi.fn(),
}));
vi.mock('../container-runner.js', async () => {
  const actual = await vi.importActual<typeof import('../container-runner.js')>(
    '../container-runner.js',
  );
  return { ...actual, getInstalledTiles: mockGetInstalledTiles };
});

import { hasIpcHandler, _resetIpcRegistryForTests } from '../ipc-registry.js';
import { registerOpsTilesIpcHandlers } from './ops-tiles.js';
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
  mockGetInstalledTiles.mockReset();
  _resetIpcRegistryForTests();
  _resetCoreIpcHandlersForTests();
});

afterAll(() => {
  fs.rmSync(path.dirname(TEST_DATA_DIR), { recursive: true, force: true });
});

describe('registerOpsTilesIpcHandlers', () => {
  it('claims all four tile-pipeline command names', () => {
    registerOpsTilesIpcHandlers();
    for (const name of [
      'promote_staging',
      'push_staged_to_branch',
      'tessl_update',
      'list_installed_tiles',
    ]) {
      expect(hasIpcHandler(name)).toBe(true);
    }
  });
});

describe('promote_staging handler', () => {
  it('refuses a non-main caller with an envelope, not silence', async () => {
    // promote_staging pushes to a tile repo with the host's GITHUB_TOKEN —
    // the isMain gate is what stops a compromised non-main container from
    // driving it by writing a task file directly. The refusal is reported
    // (#885): a silent return reads as a hang to the polling caller.
    await run(
      {
        type: 'promote_staging',
        requestId: REQUEST_ID,
        tileName: 'nanoclaw-core',
        skillName: 'status',
      },
      false,
    );
    expect(String(readEnvelope()?.error)).toContain('Only the main group');
  });

  it('rejects a tileName outside the host-side allowlist', async () => {
    // The MCP tool's zod enum mirrors this list client-side, but the IPC
    // dir is writable by any container — this set is the actual boundary,
    // and a path-shaped name is the escape it exists to stop.
    await run(
      {
        type: 'promote_staging',
        requestId: REQUEST_ID,
        tileName: '../../etc',
        skillName: 'status',
      },
      true,
    );
    const envelope = readEnvelope();
    expect(String(envelope?.error)).toContain('Invalid tileName');
    expect(String(envelope?.error)).toContain('../../etc');
  });

  it('answers a payload missing tileName with an envelope (#885)', async () => {
    // Pre-#885 the whole body was gated on all three fields, so this
    // produced no result file and the in-container runHostOperation()
    // polled until its own timeout.
    await run({ type: 'promote_staging', requestId: REQUEST_ID }, true);
    expect(String(readEnvelope()?.error)).toContain('"tileName"');
  });

  it('answers a payload missing skillName with an envelope (#885)', async () => {
    await run(
      {
        type: 'promote_staging',
        requestId: REQUEST_ID,
        tileName: 'nanoclaw-core',
      },
      true,
    );
    expect(String(readEnvelope()?.error)).toContain('"skillName"');
  });

  it('rejects a non-string tileName that skipped the MCP schema (#885)', async () => {
    // The IPC tasks dir is writable by any container, so a payload can
    // arrive without ever passing the tool's zod enum.
    await run(
      {
        type: 'promote_staging',
        requestId: REQUEST_ID,
        tileName: 42 as unknown as string,
        skillName: 'status',
      },
      true,
    );
    expect(String(readEnvelope()?.error)).toContain('"tileName"');
  });
});

describe('push_staged_to_branch handler', () => {
  it('refuses a non-main caller with an envelope', async () => {
    await run(
      {
        type: 'push_staged_to_branch',
        requestId: REQUEST_ID,
        tileName: 'nanoclaw-core',
        branch: 'fix/x',
        commitMessage: 'fix: x',
      },
      false,
    );
    expect(String(readEnvelope()?.error)).toContain('Only the main group');
  });

  it('rejects a tileName outside the allowlist', async () => {
    await run(
      {
        type: 'push_staged_to_branch',
        requestId: REQUEST_ID,
        tileName: 'not-a-tile',
        branch: 'fix/x',
        commitMessage: 'fix: x',
      },
      true,
    );
    expect(String(readEnvelope()?.error)).toContain('Invalid tileName');
  });

  it('answers a payload missing branch with an envelope (#885)', async () => {
    await run(
      {
        type: 'push_staged_to_branch',
        requestId: REQUEST_ID,
        tileName: 'nanoclaw-core',
        commitMessage: 'fix: x',
      },
      true,
    );
    expect(String(readEnvelope()?.error)).toContain('"branch"');
  });

  it('answers a payload missing commitMessage with an envelope (#885)', async () => {
    await run(
      {
        type: 'push_staged_to_branch',
        requestId: REQUEST_ID,
        tileName: 'nanoclaw-core',
        branch: 'fix/x',
      },
      true,
    );
    expect(String(readEnvelope()?.error)).toContain('"commitMessage"');
  });
});

describe('list_installed_tiles handler', () => {
  it('refuses a non-main caller with an envelope', async () => {
    await run({ type: 'list_installed_tiles', requestId: REQUEST_ID }, false);
    expect(String(readEnvelope()?.error)).toBeTruthy();
    expect(mockGetInstalledTiles).not.toHaveBeenCalled();
  });

  it('returns the registry tile list to a main caller', async () => {
    mockGetInstalledTiles.mockReturnValue(['nanoclaw-core', 'nanoclaw-travel']);
    await run({ type: 'list_installed_tiles', requestId: REQUEST_ID }, true);
    const envelope = readEnvelope();
    expect(envelope).toBeDefined();
    expect(JSON.stringify(envelope)).toContain('nanoclaw-travel');
  });
});
