import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { RegisteredGroup } from './types.js';

// cleanupOrphanNonMainHeartbeats reads getTaskById + deleteTask; keep the
// rest of db real so the module loads.
vi.mock('./db.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./db.js')>();
  return { ...actual, getTaskById: vi.fn(), deleteTask: vi.fn() };
});

import { getTaskById, deleteTask } from './db.js';
import { cleanupOrphanNonMainHeartbeats } from './group-registry.js';
import { _setRegisteredGroups } from './orchestrator-state.js';

const mockGetTaskById = vi.mocked(getTaskById);
const mockDeleteTask = vi.mocked(deleteTask);

function group(overrides: Partial<RegisteredGroup> = {}): RegisteredGroup {
  return {
    name: 'g',
    folder: 'telegram_g',
    trigger: null,
    added_at: '2024-01-01T00:00:00Z',
    ...overrides,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  _setRegisteredGroups({});
});

describe('cleanupOrphanNonMainHeartbeats', () => {
  it('deletes the heartbeat task of a non-main group that still has one', () => {
    _setRegisteredGroups({ 'g@g.us': group({ folder: 'telegram_g' }) });
    mockGetTaskById.mockReturnValue({ id: 'heartbeat-telegram_g' } as never);

    cleanupOrphanNonMainHeartbeats();

    expect(mockGetTaskById).toHaveBeenCalledWith('heartbeat-telegram_g');
    expect(mockDeleteTask).toHaveBeenCalledWith('heartbeat-telegram_g');
  });

  it('never touches a main group (main-group heartbeat is intentional)', () => {
    _setRegisteredGroups({
      'main@g.us': group({ folder: 'telegram_main', isMain: true }),
    });
    mockGetTaskById.mockReturnValue({ id: 'heartbeat-telegram_main' } as never);

    cleanupOrphanNonMainHeartbeats();

    expect(mockGetTaskById).not.toHaveBeenCalled();
    expect(mockDeleteTask).not.toHaveBeenCalled();
  });

  it('skips a non-main group that has no heartbeat task', () => {
    _setRegisteredGroups({ 'g@g.us': group({ folder: 'telegram_g' }) });
    mockGetTaskById.mockReturnValue(undefined as never);

    cleanupOrphanNonMainHeartbeats();

    expect(mockDeleteTask).not.toHaveBeenCalled();
  });
});
