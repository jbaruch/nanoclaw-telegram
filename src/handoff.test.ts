/**
 * Tests for the graceful-shutdown handoff marker (#213).
 *
 * The marker is the single source of truth for "did the previous
 * orchestrator exit gracefully" — every false-positive in either
 * direction is destructive:
 *
 *   - False "yes graceful" → cleanup-orphans skips real
 *     crash-leftovers, leaving zombie agent containers consuming
 *     resources until idle-timeout.
 *   - False "no graceful" → cleanup-orphans kills the in-flight
 *     containers a graceful prior run left behind, which is exactly
 *     the 137-cascade-on-deploy bug #213 is meant to stop.
 *
 * The reader's job is to fail closed to "no graceful" on any
 * uncertainty (parse error, schema mismatch, stale timestamp,
 * malformed shape) so the safety net is the pre-#213 cleanup
 * behaviour rather than a half-trusted marker.
 */
import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

let tempDir: string;

vi.mock('./config.js', () => ({
  // Re-evaluated per test by reassigning DATA_DIR through the
  // module's own mock. Vitest hoists vi.mock so the path needs to
  // be deterministic at import time; we redirect via a sibling
  // setter the tests update before each import.
  DATA_DIR: '/tmp/nanoclaw-handoff-test-PLACEHOLDER',
}));

vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

beforeEach(() => {
  tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-handoff-'));
  vi.resetModules();
  vi.doMock('./config.js', () => ({ DATA_DIR: tempDir }));
});

afterEach(() => {
  fs.rmSync(tempDir, { recursive: true, force: true });
});

describe('writeHandoffMarker', () => {
  it('writes a valid JSON marker with schema_version and timestamp', async () => {
    const { writeHandoffMarker } = await import('./handoff.js');

    writeHandoffMarker([
      {
        name: 'nanoclaw-foo-1',
        groupJid: 'foo@g.us',
        sessionName: 'default',
        groupFolder: 'foo',
      },
    ]);

    const raw = fs.readFileSync(path.join(tempDir, 'handoff.json'), 'utf-8');
    const parsed = JSON.parse(raw);
    expect(parsed.schema_version).toBe(1);
    // Timestamp is a real ISO-8601 string — round-trip parseable
    // and within the last few seconds of "now".
    expect(typeof parsed.shutdown_at).toBe('string');
    const ageMs = Date.now() - new Date(parsed.shutdown_at).getTime();
    expect(ageMs).toBeGreaterThanOrEqual(0);
    expect(ageMs).toBeLessThan(2000);
    expect(parsed.containers).toEqual([
      {
        name: 'nanoclaw-foo-1',
        groupJid: 'foo@g.us',
        sessionName: 'default',
        groupFolder: 'foo',
      },
    ]);
  });

  it('writes an empty container list cleanly (no active agents at shutdown)', async () => {
    const { writeHandoffMarker } = await import('./handoff.js');

    writeHandoffMarker([]);

    const parsed = JSON.parse(
      fs.readFileSync(path.join(tempDir, 'handoff.json'), 'utf-8'),
    );
    // The reader still consumes the marker — an empty containers
    // list is the explicit "graceful shutdown but nothing to adopt"
    // signal, distinct from "no marker = crash recovery".
    expect(parsed.containers).toEqual([]);
  });

  it('overwrites a prior marker atomically', async () => {
    const { writeHandoffMarker } = await import('./handoff.js');
    const markerPath = path.join(tempDir, 'handoff.json');

    writeHandoffMarker([
      {
        name: 'nanoclaw-old',
        groupJid: 'a@g.us',
        sessionName: 'default',
        groupFolder: 'a',
      },
    ]);
    writeHandoffMarker([
      {
        name: 'nanoclaw-new',
        groupJid: 'b@g.us',
        sessionName: 'default',
        groupFolder: 'b',
      },
    ]);

    const parsed = JSON.parse(fs.readFileSync(markerPath, 'utf-8'));
    // Second write wins; first marker's name is gone — important
    // because a stale "previous shutdown" payload would adopt the
    // wrong container set on the next startup.
    expect(parsed.containers).toEqual([
      {
        name: 'nanoclaw-new',
        groupJid: 'b@g.us',
        sessionName: 'default',
        groupFolder: 'b',
      },
    ]);
    // No leftover .tmp file — the rename completed.
    const leftover = fs.readdirSync(tempDir).filter((f) => f.includes('.tmp'));
    expect(leftover).toEqual([]);
  });
});

describe('readAndConsumeHandoffMarker', () => {
  it('returns null when no marker exists', async () => {
    const { readAndConsumeHandoffMarker } = await import('./handoff.js');

    expect(readAndConsumeHandoffMarker()).toBeNull();
  });

  it('returns the parsed marker AND deletes the file', async () => {
    const { writeHandoffMarker, readAndConsumeHandoffMarker } =
      await import('./handoff.js');

    writeHandoffMarker([
      {
        name: 'nanoclaw-x',
        groupJid: 'x@g.us',
        sessionName: 'default',
        groupFolder: 'x',
      },
    ]);

    const marker = readAndConsumeHandoffMarker();
    expect(marker).not.toBeNull();
    expect(marker!.containers).toHaveLength(1);
    expect(marker!.containers[0].name).toBe('nanoclaw-x');

    // File is gone — a single shutdown→startup handoff produces
    // exactly one read; leaving it would risk a future startup
    // adopting names that have long since been replaced.
    expect(fs.existsSync(path.join(tempDir, 'handoff.json'))).toBe(false);
  });

  it('returns null AND deletes the file on malformed JSON', async () => {
    const { readAndConsumeHandoffMarker } = await import('./handoff.js');
    fs.writeFileSync(path.join(tempDir, 'handoff.json'), '{not valid json');

    expect(readAndConsumeHandoffMarker()).toBeNull();
    // Deletion happens before parse so a poison marker can't trap
    // the next startup in a loop of failing to parse the same
    // bytes — caller falls through to crash-recovery cleanup.
    expect(fs.existsSync(path.join(tempDir, 'handoff.json'))).toBe(false);
  });

  it('returns null on schema_version mismatch', async () => {
    const { readAndConsumeHandoffMarker } = await import('./handoff.js');
    fs.writeFileSync(
      path.join(tempDir, 'handoff.json'),
      JSON.stringify({
        schema_version: 99,
        shutdown_at: new Date().toISOString(),
        containers: [],
      }),
    );

    // Treats unknown version as "no usable prior state" — a future
    // shape change is the trigger for the safety net.
    expect(readAndConsumeHandoffMarker()).toBeNull();
  });

  it('returns null when the marker is older than HANDOFF_TTL_MS', async () => {
    const { readAndConsumeHandoffMarker, HANDOFF_TTL_MS } =
      await import('./handoff.js');
    const old = new Date(Date.now() - HANDOFF_TTL_MS - 1000).toISOString();
    fs.writeFileSync(
      path.join(tempDir, 'handoff.json'),
      JSON.stringify({
        schema_version: 1,
        shutdown_at: old,
        containers: [
          {
            name: 'nanoclaw-stale',
            groupJid: 'a@g.us',
            sessionName: 'default',
            groupFolder: 'a',
          },
        ],
      }),
    );

    // A long-ago graceful shutdown is no different from a crash
    // for our purposes — the named containers have surely been
    // replaced or zombied since. Fall through to full cleanup.
    expect(readAndConsumeHandoffMarker()).toBeNull();
  });

  it('returns null on a future-dated marker (clock skew defense)', async () => {
    const { readAndConsumeHandoffMarker } = await import('./handoff.js');
    fs.writeFileSync(
      path.join(tempDir, 'handoff.json'),
      JSON.stringify({
        schema_version: 1,
        shutdown_at: new Date(Date.now() + 60_000).toISOString(),
        containers: [],
      }),
    );

    // Negative age implies clock jump — an attacker (or a borked
    // VM clock) shouldn't be able to extend the trust window
    // arbitrarily. Treat as untrustworthy.
    expect(readAndConsumeHandoffMarker()).toBeNull();
  });

  it('returns null when containers field is not an array', async () => {
    const { readAndConsumeHandoffMarker } = await import('./handoff.js');
    fs.writeFileSync(
      path.join(tempDir, 'handoff.json'),
      JSON.stringify({
        schema_version: 1,
        shutdown_at: new Date().toISOString(),
        containers: 'oops',
      }),
    );

    expect(readAndConsumeHandoffMarker()).toBeNull();
  });

  it('drops malformed entries (null, missing fields, wrong types) but keeps valid ones', async () => {
    const { readAndConsumeHandoffMarker } = await import('./handoff.js');
    fs.writeFileSync(
      path.join(tempDir, 'handoff.json'),
      JSON.stringify({
        schema_version: 1,
        shutdown_at: new Date().toISOString(),
        containers: [
          // Each malformed entry would otherwise crash a downstream
          // `containers.map(c => c.name)` at startup, dropping the
          // orchestrator into fail-OPEN instead of the documented
          // fail-closed-to-crash-recovery contract.
          null,
          {},
          { name: 123 }, // wrong type
          { name: '', groupJid: 'a@g.us', sessionName: 'default' }, // empty name
          { name: 'no-groupjid', sessionName: 'default' },
          {
            name: 'nanoclaw-keep',
            groupJid: 'k@g.us',
            sessionName: 'default',
            groupFolder: 'k',
          },
        ],
      }),
    );

    const marker = readAndConsumeHandoffMarker();
    expect(marker).not.toBeNull();
    // Only the well-formed entry survives; the rest are dropped
    // with a warning. The orchestrator gets a clean adoption list.
    expect(marker!.containers).toHaveLength(1);
    expect(marker!.containers[0].name).toBe('nanoclaw-keep');
  });

  it('coerces missing groupFolder to null on an otherwise-valid entry', async () => {
    const { readAndConsumeHandoffMarker } = await import('./handoff.js');
    fs.writeFileSync(
      path.join(tempDir, 'handoff.json'),
      JSON.stringify({
        schema_version: 1,
        shutdown_at: new Date().toISOString(),
        containers: [
          // groupFolder is nullable in `GroupQueue` state per the
          // existing type — the handoff round-trip must preserve
          // the same shape rather than reject the entry outright.
          {
            name: 'nanoclaw-no-folder',
            groupJid: 'x@g.us',
            sessionName: 'default',
          },
        ],
      }),
    );

    const marker = readAndConsumeHandoffMarker();
    expect(marker).not.toBeNull();
    expect(marker!.containers).toHaveLength(1);
    expect(marker!.containers[0].groupFolder).toBeNull();
  });
});

describe('handoff window (#213 Phase A spawn-collision detection)', () => {
  it('reports inactive before any markHandoffActive call', async () => {
    const { isHandoffActive, _resetHandoffWindowForTests } =
      await import('./handoff.js');
    _resetHandoffWindowForTests();

    // Steady state on a freshly-started orchestrator that didn't
    // come from a graceful shutdown — the spawn-side detector
    // must skip the `docker ps` check entirely, otherwise every
    // spawn pays a cost for nothing to look at.
    expect(isHandoffActive()).toBe(false);
  });

  it('reports active after markHandoffActive, inactive after the TTL elapses', async () => {
    vi.useFakeTimers();
    try {
      const {
        isHandoffActive,
        markHandoffActive,
        HANDOFF_TTL_MS,
        _resetHandoffWindowForTests,
      } = await import('./handoff.js');
      _resetHandoffWindowForTests();

      markHandoffActive();
      expect(isHandoffActive()).toBe(true);

      // Advance just before TTL — still active, the window is
      // load-bearing within the full TTL or a deploy with a
      // long-running agent (close to but under 5 min) would miss
      // a real collision.
      vi.advanceTimersByTime(HANDOFF_TTL_MS - 1000);
      expect(isHandoffActive()).toBe(true);

      // Advance past TTL — must clamp shut. After the window,
      // adopted containers have either exited naturally or
      // idle-timed-out, and any same-prefix collision the
      // detector might find now would be a false positive
      // against this orchestrator's own freshly-spawned containers.
      vi.advanceTimersByTime(2000);
      expect(isHandoffActive()).toBe(false);
    } finally {
      vi.useRealTimers();
    }
  });

  it('a second markHandoffActive extends the window forward', async () => {
    vi.useFakeTimers();
    try {
      const {
        isHandoffActive,
        markHandoffActive,
        HANDOFF_TTL_MS,
        _resetHandoffWindowForTests,
      } = await import('./handoff.js');
      _resetHandoffWindowForTests();

      markHandoffActive();
      vi.advanceTimersByTime(HANDOFF_TTL_MS - 10_000); // 10s before expiry
      // A second handoff (e.g. back-to-back deploys) must reset
      // the window — the new adopted containers are fresh, the
      // detector must remain active for THEIR full TTL, not
      // collapse on the prior call's expiration.
      markHandoffActive();
      vi.advanceTimersByTime(HANDOFF_TTL_MS - 1000);
      expect(isHandoffActive()).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });
});
