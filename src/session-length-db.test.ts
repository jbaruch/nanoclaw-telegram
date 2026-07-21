import { afterEach, beforeEach, describe, expect, it } from 'vitest';

import { _closeDatabase, _initTestDatabase } from './db.js';
import {
  clearSessionLengthStateForGroup,
  consumeSessionReset,
  getSessionLengthState,
  markSessionForReset,
  recordSessionTurn,
} from './db-session-length-cap.js';

// DB-helper tests for the session-length cap. Each test gets a fresh
// in-memory SQLite via `_initTestDatabase` so writes from one test
// can't leak into another (see `rules/testing-standards.md`
// independence). Fixed test data — no random IDs.

const GROUP = 'test-group-cap';
const SLOT = 'default';
const SESSION_ID_A = 'sess-aaaa-aaaa';
const SESSION_ID_B = 'sess-bbbb-bbbb';

beforeEach(() => {
  _initTestDatabase();
});

afterEach(() => {
  _closeDatabase();
});

describe('recordSessionTurn', () => {
  it('inserts a fresh row on the first turn with totals from that turn', () => {
    const row = recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 1234, null);
    expect(row.session_id).toBe(SESSION_ID_A);
    expect(row.total_input_tokens).toBe(1234);
    expect(row.turn_count).toBe(1);
    expect(row.marked_for_reset).toBe(0);
    expect(row.last_handoff_summary).toBeNull();
    expect(row.started_at).toBeTruthy();
    expect(row.last_updated_at).toBeTruthy();
  });

  it('accumulates tokens and increments turn_count on subsequent same-session turns', () => {
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 100, null);
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 200, null);
    const row = recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 300, null);
    expect(row.total_input_tokens).toBe(600);
    expect(row.turn_count).toBe(3);
  });

  it('preserves last_handoff_summary across turns via COALESCE (write-once)', () => {
    recordSessionTurn(
      GROUP,
      SLOT,
      SESSION_ID_A,
      50,
      '<session-handoff>x</session-handoff>',
    );
    // Subsequent turn passes null — must NOT overwrite the prefix.
    const row = recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 50, null);
    expect(row.last_handoff_summary).toBe(
      '<session-handoff>x</session-handoff>',
    );
  });

  it('replaces the row (not accumulates) when session_id changes underneath us', () => {
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 5_000, null);
    // SDK chain rotated without going through the cap-state reset
    // path — defensive shape: treat as a fresh session.
    const row = recordSessionTurn(GROUP, SLOT, SESSION_ID_B, 100, null);
    expect(row.session_id).toBe(SESSION_ID_B);
    expect(row.total_input_tokens).toBe(100);
    expect(row.turn_count).toBe(1);
  });

  it('keeps separate accounting for default and maintenance slots in the same group', () => {
    recordSessionTurn(GROUP, 'default', SESSION_ID_A, 10, null);
    recordSessionTurn(GROUP, 'maintenance', SESSION_ID_B, 99, null);
    const def = getSessionLengthState(GROUP, 'default');
    const main = getSessionLengthState(GROUP, 'maintenance');
    expect(def?.total_input_tokens).toBe(10);
    expect(main?.total_input_tokens).toBe(99);
    expect(def?.session_id).toBe(SESSION_ID_A);
    expect(main?.session_id).toBe(SESSION_ID_B);
  });
});

describe('markSessionForReset', () => {
  it('sets marked_for_reset and persists reason + cap', () => {
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 1_000_000, null);
    const changed = markSessionForReset(GROUP, SLOT, 'token_cap', 1_000_000);
    expect(changed).toBe(1);
    const row = getSessionLengthState(GROUP, SLOT);
    expect(row?.marked_for_reset).toBe(1);
    expect(row?.reset_reason).toBe('token_cap');
    expect(row?.reset_cap).toBe(1_000_000);
  });

  it('is idempotent — a second mark on the same row reports zero changes', () => {
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 1_000_000, null);
    expect(markSessionForReset(GROUP, SLOT, 'token_cap', 1_000_000)).toBe(1);
    // Second mark with a DIFFERENT reason must not overwrite (the
    // first reason is the actual trip, not the second).
    expect(markSessionForReset(GROUP, SLOT, 'turn_cap', 100)).toBe(0);
    const row = getSessionLengthState(GROUP, SLOT);
    expect(row?.reset_reason).toBe('token_cap');
    expect(row?.reset_cap).toBe(1_000_000);
  });

  it('returns 0 when no row exists for the slot', () => {
    expect(
      markSessionForReset('nonexistent', SLOT, 'token_cap', 1_000_000),
    ).toBe(0);
  });
});

describe('consumeSessionReset', () => {
  it('returns null when no reset is pending', () => {
    expect(consumeSessionReset(GROUP, SLOT)).toBeNull();
  });

  it('returns null when row exists but is not marked for reset', () => {
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 1, null);
    expect(consumeSessionReset(GROUP, SLOT)).toBeNull();
  });

  it('reads-and-deletes atomically when a reset is pending', () => {
    recordSessionTurn(
      GROUP,
      SLOT,
      SESSION_ID_A,
      1_500_000,
      '<session-handoff>prior work</session-handoff>',
    );
    markSessionForReset(GROUP, SLOT, 'token_cap', 1_000_000);

    const consumed = consumeSessionReset(GROUP, SLOT);
    expect(consumed).toEqual({
      sessionId: SESSION_ID_A,
      lastHandoffSummary: '<session-handoff>prior work</session-handoff>',
      reason: 'token_cap',
      cap: 1_000_000,
    });
    // Row must be gone after consume.
    expect(getSessionLengthState(GROUP, SLOT)).toBeUndefined();
  });

  it('surfaces turn_cap reason verbatim when that was the trigger', () => {
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 100, null);
    markSessionForReset(GROUP, SLOT, 'turn_cap', 100);
    const consumed = consumeSessionReset(GROUP, SLOT);
    expect(consumed?.reason).toBe('turn_cap');
    expect(consumed?.cap).toBe(100);
  });

  it('does not double-consume — a second call after consume returns null', () => {
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 1, null);
    markSessionForReset(GROUP, SLOT, 'token_cap', 1);
    expect(consumeSessionReset(GROUP, SLOT)).not.toBeNull();
    expect(consumeSessionReset(GROUP, SLOT)).toBeNull();
  });
});

describe('clearSessionLengthStateForGroup', () => {
  it('removes every slot for the group and reports the count', () => {
    recordSessionTurn(GROUP, 'default', SESSION_ID_A, 1, null);
    recordSessionTurn(GROUP, 'maintenance', SESSION_ID_B, 1, null);
    recordSessionTurn('other-group', 'default', SESSION_ID_A, 1, null);

    expect(clearSessionLengthStateForGroup(GROUP)).toBe(2);
    expect(getSessionLengthState(GROUP, 'default')).toBeUndefined();
    expect(getSessionLengthState(GROUP, 'maintenance')).toBeUndefined();
    // Other group untouched.
    expect(getSessionLengthState('other-group', 'default')).toBeDefined();
  });

  it('reports 0 when no rows exist for the group (no-op tolerated)', () => {
    expect(clearSessionLengthStateForGroup('never-seen')).toBe(0);
  });
});

describe('integration: lifecycle from first turn to reset to fresh row', () => {
  it('full cycle: accumulate → mark → consume → next-turn re-INSERT under new sessionId', () => {
    // 1. First turn.
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 200_000, null);
    // 2. Accumulate to threshold over multiple turns.
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 300_000, null);
    recordSessionTurn(GROUP, SLOT, SESSION_ID_A, 500_000, null);
    const beforeMark = getSessionLengthState(GROUP, SLOT);
    expect(beforeMark?.total_input_tokens).toBe(1_000_000);
    expect(beforeMark?.turn_count).toBe(3);

    // 3. Threshold check fires after the third turn.
    markSessionForReset(GROUP, SLOT, 'token_cap', 1_000_000);

    // 4. Next inbound consumes the reset. Row deleted.
    const consumed = consumeSessionReset(GROUP, SLOT);
    expect(consumed?.sessionId).toBe(SESSION_ID_A);
    expect(getSessionLengthState(GROUP, SLOT)).toBeUndefined();

    // 5. Fresh session's first turn re-INSERTs cleanly.
    recordSessionTurn(
      GROUP,
      SLOT,
      SESSION_ID_B,
      50_000,
      '<session-handoff>continued</session-handoff>',
    );
    const fresh = getSessionLengthState(GROUP, SLOT);
    expect(fresh?.session_id).toBe(SESSION_ID_B);
    expect(fresh?.total_input_tokens).toBe(50_000);
    expect(fresh?.turn_count).toBe(1);
    expect(fresh?.marked_for_reset).toBe(0);
    expect(fresh?.last_handoff_summary).toBe(
      '<session-handoff>continued</session-handoff>',
    );
  });
});
