import { describe, it, expect } from 'vitest';

import {
  buildRegisterGroupContainerConfig,
  describeOverlayUpdate,
} from './overlay-tiles.js';

// #305 follow-up — the bridge's `register_group` and
// `set_additional_tiles` MCP tools delegate `containerConfig`
// assembly and operator-message rendering to these helpers. The
// host-side IPC handlers carry the actual mutation logic and live
// under test in `src/ipc-auth.test.ts`; these tests pin the shape
// the bridge sends ON THE WIRE so a regression in the bridge can't
// silently mis-shape the IPC payload.

describe('buildRegisterGroupContainerConfig (#305)', () => {
  it('returns undefined when no contributing field is set', () => {
    expect(buildRegisterGroupContainerConfig({})).toBeUndefined();
  });

  it('omits trusted when undefined (so the host-side default applies)', () => {
    expect(
      buildRegisterGroupContainerConfig({ enableHeartbeat: true }),
    ).toEqual({ enableHeartbeat: true });
  });

  it('forwards trusted=true', () => {
    expect(buildRegisterGroupContainerConfig({ trusted: true })).toEqual({
      trusted: true,
    });
  });

  it('forwards trusted=false explicitly (untrusted opt-in distinct from absent)', () => {
    expect(buildRegisterGroupContainerConfig({ trusted: false })).toEqual({
      trusted: false,
    });
  });

  it('forwards additionalMounts verbatim', () => {
    const mounts = [
      { hostPath: '/tmp/x', containerPath: 'extra', readonly: true },
    ];
    expect(
      buildRegisterGroupContainerConfig({ additionalMounts: mounts }),
    ).toEqual({ additionalMounts: mounts });
  });

  it('forwards additionalTiles when non-empty', () => {
    expect(
      buildRegisterGroupContainerConfig({
        additionalTiles: ['nanoclaw-coding'],
      }),
    ).toEqual({ additionalTiles: ['nanoclaw-coding'] });
  });

  it('OMITS additionalTiles when the array is empty (treats [] as "no overlay")', () => {
    // Empty array shouldn't spuriously create a containerConfig
    // object — the host treats undefined and "absent additionalTiles"
    // the same way, but `containerConfig: {}` vs `containerConfig:
    // undefined` differ on the untrusted-by-default semantics. So
    // when ONLY additionalTiles is given and it's [], the entire
    // containerConfig stays undefined.
    expect(
      buildRegisterGroupContainerConfig({ additionalTiles: [] }),
    ).toBeUndefined();
  });

  it('combines all four fields preserving each when set', () => {
    const cfg = buildRegisterGroupContainerConfig({
      trusted: true,
      enableHeartbeat: true,
      additionalMounts: [{ hostPath: '/tmp/x', readonly: true }],
      additionalTiles: ['nanoclaw-coding', 'nanoclaw-family'],
    });
    expect(cfg).toEqual({
      trusted: true,
      enableHeartbeat: true,
      additionalMounts: [{ hostPath: '/tmp/x', readonly: true }],
      additionalTiles: ['nanoclaw-coding', 'nanoclaw-family'],
    });
  });

  it('builds a config when ONLY additionalTiles is non-empty (the new #305 path)', () => {
    // Pre-#305-follow-up the assembly didn't include additionalTiles
    // in the gating predicate, so an `additionalTiles`-only call
    // would have returned undefined and dropped the overlay before
    // the IPC ever left the container. This test pins that the new
    // gating predicate honours additionalTiles too.
    const cfg = buildRegisterGroupContainerConfig({
      additionalTiles: ['nanoclaw-coding'],
    });
    expect(cfg).toEqual({ additionalTiles: ['nanoclaw-coding'] });
  });
});

describe('describeOverlayUpdate (#305)', () => {
  it('renders null as "cleared (baseline only)"', () => {
    expect(describeOverlayUpdate(null)).toBe('cleared (baseline only)');
  });

  it('renders empty array identically to null (host treats them the same)', () => {
    expect(describeOverlayUpdate([])).toBe('cleared (baseline only)');
  });

  it('renders a single overlay tile', () => {
    expect(describeOverlayUpdate(['nanoclaw-coding'])).toBe(
      '[nanoclaw-coding]',
    );
  });

  it('renders multiple overlay tiles comma-separated in the configured order', () => {
    expect(describeOverlayUpdate(['nanoclaw-coding', 'nanoclaw-family'])).toBe(
      '[nanoclaw-coding, nanoclaw-family]',
    );
  });
});
