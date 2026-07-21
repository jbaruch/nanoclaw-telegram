import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { RegisteredGroup } from './types.js';

// findChannel is the only router export the tested guard paths touch;
// keep the rest real so the module loads normally.
vi.mock('./router.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./router.js')>();
  return { ...actual, findChannel: vi.fn() };
});

import { recordFailure } from './circuit-breaker.js';
import { processGroupMessages } from './message-pipeline.js';
import { channels } from './orchestrator-runtime.js';
import { _setRegisteredGroups } from './orchestrator-state.js';
import { findChannel } from './router.js';

const mockFindChannel = vi.mocked(findChannel);

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
  channels.length = 0; // shared runtime singleton — reset between tests
});

afterEach(() => {
  channels.length = 0;
});

describe('processGroupMessages — spawn guards', () => {
  it('skips (returns true) when the JID has no registered group', async () => {
    _setRegisteredGroups({});
    await expect(processGroupMessages('unknown@g.us')).resolves.toBe(true);
    // Never even looked for a channel.
    expect(mockFindChannel).not.toHaveBeenCalled();
  });

  it('skips (returns true) when no channel owns the JID', async () => {
    _setRegisteredGroups({ 'g@g.us': group() });
    mockFindChannel.mockReturnValue(undefined);
    await expect(processGroupMessages('g@g.us')).resolves.toBe(true);
    expect(mockFindChannel).toHaveBeenCalledWith(channels, 'g@g.us');
  });

  it('skips (returns true) while the circuit breaker is tripped for the folder', async () => {
    _setRegisteredGroups({ 'c@g.us': group({ folder: 'telegram_breaker' }) });
    // A channel owns the JID, so the breaker gate (not the channel gate)
    // is what short-circuits.
    mockFindChannel.mockReturnValue({} as never);
    // Trip the breaker: five consecutive failures arm the cooldown.
    for (let i = 0; i < 5; i += 1) recordFailure('telegram_breaker');
    await expect(processGroupMessages('c@g.us')).resolves.toBe(true);
  });
});
