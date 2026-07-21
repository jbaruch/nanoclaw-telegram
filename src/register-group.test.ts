// #670 — register_group's IPC payload carries only the `trigger` string,
// never a `triggerPatterns` config. `registerGroup` persists it (the
// serializer synthesizes the keyword column) and must then cache the
// HYDRATED group, not the raw payload — the trigger gate reads
// `triggerPatterns` from the in-memory registry and would otherwise see
// "no patterns configured" and fail-open on every message until reload.
import { describe, it, expect, beforeEach } from 'vitest';

import { hydrateRegisteredGroupTriggerPatterns } from './index.js';
import { _initTestDatabase } from './db.js';
import {
  getTriggerPatterns,
  setRegisteredGroup,
} from './db-registered-groups.js';
import type { RegisteredGroup } from './types.js';

describe('hydrateRegisteredGroupTriggerPatterns (#670)', () => {
  beforeEach(() => {
    _initTestDatabase();
  });

  it('repopulates triggerPatterns from the persisted row when the payload omits it', () => {
    const payload: RegisteredGroup = {
      name: 'WTF',
      folder: 'telegram_wtf',
      trigger: '@AyeAye',
      added_at: '2024-01-01T00:00:00.000Z',
      requiresTrigger: true,
    };
    // The register_group IPC payload never carries a triggerPatterns config.
    expect(payload.triggerPatterns).toBeUndefined();

    // registerGroup persists the payload, then caches the hydrated object.
    setRegisteredGroup('wtf@g.us', payload);
    const cached = hydrateRegisteredGroupTriggerPatterns(payload, 'wtf@g.us');

    expect(cached.triggerPatterns).toBeDefined();
    expect(cached.triggerPatterns!.patterns).toHaveLength(1);
    expect(cached.triggerPatterns!.patterns[0]).toMatchObject({
      pattern: '@AyeAye',
      kind: 'keyword',
      source: 'owner-set',
    });
    // The DB row the hydration reads from agrees.
    expect(getTriggerPatterns('wtf@g.us')!.patterns[0].pattern).toBe('@AyeAye');
    // Every other field of the payload is preserved.
    expect(cached.trigger).toBe('@AyeAye');
    expect(cached.requiresTrigger).toBe(true);
    expect(cached.folder).toBe('telegram_wtf');
  });
});
