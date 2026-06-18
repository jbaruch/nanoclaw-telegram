import { describe, expect, it } from 'vitest';

import {
  buildSetSessionCapsPayload,
  describeSessionCapsChange,
  isEmptySessionCapsUpdate,
} from './session-caps-payload.js';

// Fixed-data tests per `rules/testing-standards.md`. The MCP server
// registration in `ipc-mcp-stdio.ts` is a thin adapter over these pure
// helpers; testing them here pins the empty-update guard and payload
// shape without standing up the MCP server.
const NOW = new Date('2026-06-18T12:00:00.000Z');

describe('isEmptySessionCapsUpdate', () => {
  it('is true when neither cap is provided (the host-rejected no-op)', () => {
    expect(isEmptySessionCapsUpdate({ groupFolder: 'g' })).toBe(true);
  });

  it('is false when a turn cap is provided', () => {
    expect(
      isEmptySessionCapsUpdate({ groupFolder: 'g', sessionTurnCap: 40 }),
    ).toBe(false);
  });

  it('is false when a cap is explicitly cleared with null', () => {
    // null is a real "clear" request, not an omission.
    expect(
      isEmptySessionCapsUpdate({ groupFolder: 'g', sessionTurnCap: null }),
    ).toBe(false);
  });

  it('is false when only the token cap is provided', () => {
    expect(
      isEmptySessionCapsUpdate({ groupFolder: 'g', sessionTokenCap: 500_000 }),
    ).toBe(false);
  });
});

describe('buildSetSessionCapsPayload', () => {
  it('carries only the explicitly-provided caps (omit = leave unchanged)', () => {
    expect(
      buildSetSessionCapsPayload({ groupFolder: 'g', sessionTurnCap: 40 }, NOW),
    ).toEqual({
      type: 'set_session_caps',
      groupFolder: 'g',
      sessionTurnCap: 40,
      timestamp: '2026-06-18T12:00:00.000Z',
    });
  });

  it('preserves an explicit null (clear) distinct from omission', () => {
    const payload = buildSetSessionCapsPayload(
      { groupFolder: 'g', sessionTurnCap: null },
      NOW,
    );
    expect(payload.sessionTurnCap).toBeNull();
    expect('sessionTokenCap' in payload).toBe(false);
  });

  it('carries both caps when both provided', () => {
    expect(
      buildSetSessionCapsPayload(
        { groupFolder: 'g', sessionTurnCap: 40, sessionTokenCap: 250_000 },
        NOW,
      ),
    ).toEqual({
      type: 'set_session_caps',
      groupFolder: 'g',
      sessionTurnCap: 40,
      sessionTokenCap: 250_000,
      timestamp: '2026-06-18T12:00:00.000Z',
    });
  });
});

describe('describeSessionCapsChange', () => {
  it('describes a set value', () => {
    expect(
      describeSessionCapsChange(
        buildSetSessionCapsPayload(
          { groupFolder: 'g', sessionTurnCap: 40 },
          NOW,
        ),
      ),
    ).toBe('turn cap → 40');
  });

  it('describes a clear', () => {
    expect(
      describeSessionCapsChange(
        buildSetSessionCapsPayload(
          { groupFolder: 'g', sessionTokenCap: null },
          NOW,
        ),
      ),
    ).toBe('token cap cleared (use global default)');
  });

  it('joins both changes', () => {
    expect(
      describeSessionCapsChange(
        buildSetSessionCapsPayload(
          { groupFolder: 'g', sessionTurnCap: 40, sessionTokenCap: null },
          NOW,
        ),
      ),
    ).toBe('turn cap → 40, token cap cleared (use global default)');
  });
});
