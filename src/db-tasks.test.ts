import { describe, expect, it } from 'vitest';

import { rebuildCadenceRegistryForGroup } from './db-tasks.js';

// This file must NOT call _initTestDatabase(): the pre-init guard under
// test reads db-connection's registration flag, and vitest's per-file
// module isolation guarantees a fresh (unregistered) db-connection here
// regardless of what other test files do.
describe('rebuildCadenceRegistryForGroup pre-init guard', () => {
  it('throws the function-specific error, not the generic sentinel, before initDatabase', () => {
    expect(() =>
      rebuildCadenceRegistryForGroup({
        groupFolder: 'test-group',
        chatJid: 'tg:123',
        createdByRole: 'owner',
        skillsDir: '/nonexistent/skills',
        computeNextRun: () => '2026-01-01T00:00:00.000Z',
        now: () => new Date('2026-01-01T00:00:00.000Z'),
      }),
    ).toThrow(/rebuildCadenceRegistryForGroup called before initDatabase/);
  });
});
