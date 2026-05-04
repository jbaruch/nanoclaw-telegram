import { describe, it, expect, beforeEach } from 'vitest';

import {
  registerContainer,
  lookupContainer,
  unregisterContainer,
  generateToken,
  _resetRegistry,
  _registrySize,
} from './proxy-registry.js';

describe('proxy-registry', () => {
  beforeEach(() => _resetRegistry());

  it('register → lookup roundtrips context', () => {
    const ctx = {
      group: 'telegram_main',
      tier: 'main' as const,
      session: 'default',
      task_id: null,
      message_id: null,
    };
    const token = registerContainer(ctx);
    expect(typeof token).toBe('string');
    expect(token.length).toBeGreaterThan(10);
    expect(lookupContainer(token)).toEqual(ctx);
  });

  it('unregister removes the entry', () => {
    const token = registerContainer({
      group: 'g',
      tier: 'untrusted',
      session: 's',
      task_id: null,
      message_id: null,
    });
    expect(_registrySize()).toBe(1);
    unregisterContainer(token);
    expect(_registrySize()).toBe(0);
    expect(lookupContainer(token)).toBeNull();
  });

  it('unknown tokens lookup to null', () => {
    expect(lookupContainer('nonsense')).toBeNull();
  });

  it('generated tokens are URL-safe and 16-byte base64url (~22 chars)', () => {
    // Determinism: assert STABLE properties of the encoding (URL-safe
    // alphabet + the length implied by 16 random bytes encoded as
    // base64url) instead of probabilistic uniqueness — two random
    // 16-byte draws colliding is astronomically unlikely but not zero.
    const t = generateToken();
    expect(t).toMatch(/^[A-Za-z0-9_-]+$/);
    // 16 bytes → 22 chars of base64url (no padding from
    // base64url-encoded `randomBytes(16)` per Node docs).
    expect(t.length).toBe(22);
  });
});
