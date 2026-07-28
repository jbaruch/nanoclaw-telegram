import { describe, it, expect } from 'vitest';

import {
  buildTimeoutKillAlert,
  shouldAlertTimeoutKill,
} from './timeout-kill-alert.js';

describe('shouldAlertTimeoutKill (#890 follow-up)', () => {
  // `timedOut` alone is the predicate — the emitters already decide
  // what counts as a kill. Notably a container that idles out AFTER
  // delivering its result is never stamped, because nothing was killed
  // mid-work and that is how every healthy maintenance run ends.

  it('alerts on a timeout kill', () => {
    expect(shouldAlertTimeoutKill(true)).toBe(true);
  });

  it('stays silent when the run was not a timeout kill', () => {
    expect(shouldAlertTimeoutKill(false)).toBe(false);
  });
});

describe('buildTimeoutKillAlert (#890 follow-up)', () => {
  it('names the skill, the task, the duration and the recorded status', () => {
    const text = buildTimeoutKillAlert({
      taskId: 'cadence-registry::grp::tessl__flight-assist',
      skillName: 'tessl__flight-assist',
      durationMs: 30_412,
      runStatus: 'error',
      error: 'precheck script failed: execfile-error (timed out after 30s)',
    });
    expect(text).toContain('tessl__flight-assist');
    expect(text).toContain('cadence-registry::grp::tessl__flight-assist');
    expect(text).toContain('30.4s');
    expect(text).toContain('error');
    expect(text).toContain('timed out after 30s');
  });

  it('falls back to the task id when no skill was resolved', () => {
    // An ad-hoc scheduled task whose prompt names no skill.
    const text = buildTimeoutKillAlert({
      taskId: 'once::nightly-sweep',
      durationMs: 1_000,
      runStatus: 'killed',
    });
    expect(text).toContain('once::nightly-sweep');
    expect(text).not.toContain('undefined');
  });

  it('carries the container-kill reason', () => {
    const text = buildTimeoutKillAlert({
      taskId: 'cadence-registry::grp::tessl__drive-engine',
      skillName: 'tessl__drive-engine',
      durationMs: 300_000,
      runStatus: 'killed',
      error:
        'Maintenance container reaped by inactivity timeout after 300000ms having streamed only preview output and no terminal result — incomplete run (reaped mid-compose), retriable',
    });
    expect(text).toContain('reaped by inactivity timeout');
    expect(text).toContain('300.0s');
  });

  it('truncates a long reason so one alert cannot flood the chat', () => {
    const text = buildTimeoutKillAlert({
      taskId: 't',
      durationMs: 1,
      runStatus: 'error',
      error: 'x'.repeat(5_000),
    });
    expect(text.length).toBeLessThan(500);
    expect(text).toContain('…');
  });

  it('omits the reason line when there is no error text', () => {
    const text = buildTimeoutKillAlert({
      taskId: 't',
      durationMs: 2_500,
      runStatus: 'killed',
      error: null,
    });
    expect(text.split('\n')).toHaveLength(2);
    expect(text).toContain('2.5s');
  });

  it('adds no channel-specific markup of its own', () => {
    // The alert crosses whichever channel the main group is on, so
    // markdown/HTML added by the builder would render as literal
    // characters on the others. Underscores inside a skill name are
    // data the caller supplied, not markup this builder introduced —
    // so the check is on the decoration, not on every symbol.
    const text = buildTimeoutKillAlert({
      taskId: 'cadence-registry::grp::tessl__flight-assist',
      skillName: 'tessl__flight-assist',
      durationMs: 30_000,
      runStatus: 'error',
      error: 'timed out',
    });
    expect(text).not.toMatch(/<\/?[a-z]+>/i);
    expect(text).not.toContain('`');
    expect(text).not.toMatch(/\*\*/);
  });
});
