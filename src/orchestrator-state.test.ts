import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('./db.js', () => ({
  getRouterState: vi.fn(),
  setRouterState: vi.fn(),
  getAllSessions: vi.fn(() => ({})),
  getAllRegisteredGroups: vi.fn(() => ({})),
  getLastBotMessageTimestamp: vi.fn(),
}));

import {
  getAllRegisteredGroups,
  getAllSessions,
  getLastBotMessageTimestamp,
  getRouterState,
  setRouterState,
} from './db.js';
import * as state from './orchestrator-state.js';

const mockGetRouterState = vi.mocked(getRouterState);
const mockSetRouterState = vi.mocked(setRouterState);
const mockGetAllSessions = vi.mocked(getAllSessions);
const mockGetAllRegisteredGroups = vi.mocked(getAllRegisteredGroups);
const mockGetLastBotMessageTimestamp = vi.mocked(getLastBotMessageTimestamp);

beforeEach(() => {
  vi.clearAllMocks();
  mockGetRouterState.mockReturnValue(undefined as never);
  mockGetAllSessions.mockReturnValue({});
  mockGetAllRegisteredGroups.mockReturnValue({});
  mockGetLastBotMessageTimestamp.mockReturnValue(undefined as never);
  // Reset EVERY exported singleton to a known baseline by loading from the
  // (empty) mocked stores — clears lastTimestamp, lastAgentTimestamp,
  // sessions, AND registeredGroups so no state leaks between tests
  // (jbaruch/coding-policy: testing-standards — independence).
  state.loadState();
});

describe('loadState', () => {
  it('hydrates all four singletons from the router/session/group stores', () => {
    mockGetRouterState.mockImplementation((key: string) =>
      key === 'last_timestamp'
        ? '2026-01-01T00:00:00Z'
        : '{"g@g.us":"2026-01-01T00:05:00Z"}',
    );
    mockGetAllSessions.mockReturnValue({ telegram_g: { default: 'sess-1' } });
    mockGetAllRegisteredGroups.mockReturnValue({
      'g@g.us': {
        name: 'g',
        folder: 'telegram_g',
        trigger: null,
        added_at: 'x',
      },
    });

    state.loadState();

    expect(state.lastTimestamp).toBe('2026-01-01T00:00:00Z');
    expect(state.lastAgentTimestamp).toEqual({
      'g@g.us': '2026-01-01T00:05:00Z',
    });
    expect(state.sessions).toEqual({ telegram_g: { default: 'sess-1' } });
    expect(state.registeredGroups['g@g.us']?.folder).toBe('telegram_g');
  });

  it('resets lastAgentTimestamp to {} on corrupted JSON (SyntaxError swallowed)', () => {
    mockGetRouterState.mockImplementation((key: string) =>
      key === 'last_agent_timestamp' ? '{not valid json' : '',
    );

    state.loadState();

    expect(state.lastAgentTimestamp).toEqual({});
  });

  it('defaults to empty state when the router keys are unset', () => {
    mockGetRouterState.mockReturnValue(undefined as never);
    state.loadState();
    expect(state.lastTimestamp).toBe('');
    expect(state.lastAgentTimestamp).toEqual({});
  });
});

describe('saveState', () => {
  it('persists the current lastTimestamp and lastAgentTimestamp', () => {
    mockGetRouterState.mockImplementation((key: string) =>
      key === 'last_timestamp'
        ? '2026-02-02T02:02:02Z'
        : '{"g@g.us":"2026-02-02T03:03:03Z"}',
    );
    state.loadState();
    mockSetRouterState.mockClear();

    state.saveState();

    expect(mockSetRouterState).toHaveBeenCalledWith(
      'last_timestamp',
      '2026-02-02T02:02:02Z',
    );
    expect(mockSetRouterState).toHaveBeenCalledWith(
      'last_agent_timestamp',
      '{"g@g.us":"2026-02-02T03:03:03Z"}',
    );
  });
});

describe('getOrRecoverCursor', () => {
  it('returns the existing cursor without touching the bot-message store', () => {
    mockGetRouterState.mockImplementation((key: string) =>
      key === 'last_agent_timestamp' ? '{"g@g.us":"2026-03-03T00:00:00Z"}' : '',
    );
    state.loadState();

    expect(state.getOrRecoverCursor('g@g.us')).toBe('2026-03-03T00:00:00Z');
    expect(mockGetLastBotMessageTimestamp).not.toHaveBeenCalled();
  });

  it('recovers from the last bot reply, records it, and persists', () => {
    mockGetLastBotMessageTimestamp.mockReturnValue('2026-04-04T00:00:00Z');

    const cursor = state.getOrRecoverCursor('new@g.us');

    expect(cursor).toBe('2026-04-04T00:00:00Z');
    expect(state.lastAgentTimestamp['new@g.us']).toBe('2026-04-04T00:00:00Z');
    expect(mockSetRouterState).toHaveBeenCalledWith(
      'last_agent_timestamp',
      expect.stringContaining('2026-04-04T00:00:00Z'),
    );
  });

  it('returns empty string when there is no cursor and no bot reply', () => {
    mockGetLastBotMessageTimestamp.mockReturnValue(undefined as never);
    expect(state.getOrRecoverCursor('barren@g.us')).toBe('');
  });
});

describe('_setRegisteredGroups', () => {
  it('replaces the registeredGroups binding wholesale', () => {
    state._setRegisteredGroups({
      'a@g.us': {
        name: 'a',
        folder: 'telegram_a',
        trigger: null,
        added_at: 'x',
      },
    });
    expect(Object.keys(state.registeredGroups)).toEqual(['a@g.us']);
  });
});
