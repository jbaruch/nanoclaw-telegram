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
    `nanoclaw-ops-git-test-${process.pid}`,
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

// The git pipelines themselves are covered in `git-persist.test.ts` and
// `backup-sync.test.ts` against real throwaway repos. These tests pin the
// HANDLER contract — registration, the isMain gate, payload validation,
// and the result-envelope shape — so a stub is the right depth here.
const { mockSyncBackupRepo, mockRunSerializedPersonaPersist } = vi.hoisted(
  () => ({
    mockSyncBackupRepo: vi.fn(),
    mockRunSerializedPersonaPersist: vi.fn(),
  }),
);
vi.mock('../backup-sync.js', () => ({ syncBackupRepo: mockSyncBackupRepo }));
vi.mock('../git-persist.js', async () => {
  const actual =
    await vi.importActual<typeof import('../git-persist.js')>(
      '../git-persist.js',
    );
  return {
    ...actual,
    runSerializedPersonaPersist: mockRunSerializedPersonaPersist,
  };
});

import { hasIpcHandler, _resetIpcRegistryForTests } from '../ipc-registry.js';
import { registerOpsGitIpcHandlers } from './ops-git.js';
import { processTaskIpc } from '../ipc.js';
import { _resetCoreIpcHandlersForTests } from './index.js';
import type { IpcDeps } from '../ipc.js';
import type { IpcTaskPayload } from '../ipc-registry.js';

const SOURCE_GROUP = 'main-group';
const REQUEST_ID = 'req-1';

/** Where the handler writes its reply envelope for REQUEST_ID. */
function resultPath(): string {
  return path.join(
    TEST_DATA_DIR,
    'ipc',
    SOURCE_GROUP,
    'input-default',
    `_script_result_${REQUEST_ID}.json`,
  );
}

function readEnvelope(): Record<string, unknown> | undefined {
  const p = resultPath();
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
  mockSyncBackupRepo.mockReset();
  mockRunSerializedPersonaPersist.mockReset();
  _resetIpcRegistryForTests();
  _resetCoreIpcHandlersForTests();
});

afterAll(() => {
  fs.rmSync(path.dirname(TEST_DATA_DIR), { recursive: true, force: true });
});

describe('registerOpsGitIpcHandlers', () => {
  it('claims both git-backed command names', () => {
    registerOpsGitIpcHandlers();
    expect(hasIpcHandler('github_backup')).toBe(true);
    expect(hasIpcHandler('persist_global_file')).toBe(true);
  });
});

describe('github_backup handler', () => {
  it('refuses a non-main caller and never touches the backup pipeline', async () => {
    await run({ type: 'github_backup', requestId: REQUEST_ID }, false);
    // Host-side gate: a compromised non-main container must not be able
    // to drive the GITHUB_TOKEN-bearing sync by writing a task file.
    expect(mockSyncBackupRepo).not.toHaveBeenCalled();
    expect(readEnvelope()).toBeUndefined();
  });

  it('does nothing at all without a requestId', async () => {
    // The whole handler body is requestId-gated — a fire-and-forget
    // payload has nowhere to send a reply, so it is a no-op even for main.
    await run({ type: 'github_backup' }, true);
    expect(mockSyncBackupRepo).not.toHaveBeenCalled();
  });

  it('converts a sync failure into a structured envelope rather than throwing', async () => {
    mockSyncBackupRepo.mockImplementation(() => {
      throw new Error('backup-repo missing');
    });
    await run({ type: 'github_backup', requestId: REQUEST_ID }, true);
    const envelope = readEnvelope();
    expect(envelope).toBeDefined();
    expect(String(envelope?.error)).toContain('backup-repo missing');
    expect(envelope?.stage).toBe('sync');
  });

  it('propagates a non-Error throw instead of masking a programming bug', async () => {
    mockSyncBackupRepo.mockImplementation(() => {
      throw 'not-an-error';
    });
    await expect(
      run({ type: 'github_backup', requestId: REQUEST_ID }, true),
    ).rejects.toBe('not-an-error');
  });
});

describe('persist_global_file handler', () => {
  it('refuses a non-main caller and never queues a persist', async () => {
    await run({ type: 'persist_global_file', requestId: REQUEST_ID }, false);
    expect(mockRunSerializedPersonaPersist).not.toHaveBeenCalled();
    expect(readEnvelope()).toBeUndefined();
  });

  it('rejects a non-allowlisted file with a validate-stage envelope', async () => {
    await run(
      {
        type: 'persist_global_file',
        requestId: REQUEST_ID,
        files: ['../../.env'],
      },
      true,
    );
    const envelope = readEnvelope();
    expect(envelope?.stage).toBe('validate');
    expect(String(envelope?.error)).toContain('../../.env');
    // Rejected before anything reached the git pipeline.
    expect(mockRunSerializedPersonaPersist).not.toHaveBeenCalled();
  });

  it('rejects a tracked-but-not-persona file the same way', async () => {
    await run(
      {
        type: 'persist_global_file',
        requestId: REQUEST_ID,
        files: ['CLAUDE.md'],
      },
      true,
    );
    expect(readEnvelope()?.stage).toBe('validate');
    expect(mockRunSerializedPersonaPersist).not.toHaveBeenCalled();
  });

  it('queues a serialized persist for an allowlisted file', async () => {
    await run(
      {
        type: 'persist_global_file',
        requestId: REQUEST_ID,
        files: ['SOUL.md'],
      },
      true,
    );
    expect(mockRunSerializedPersonaPersist).toHaveBeenCalledOnce();
    // Serialization is the point: concurrent persists would race on the
    // shared clone (see `runSerializedPersonaPersist`).
    expect(typeof mockRunSerializedPersonaPersist.mock.calls[0][0]).toBe(
      'function',
    );
  });
});
