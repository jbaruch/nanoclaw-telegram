import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

import { afterEach, describe, expect, it } from 'vitest';

import { runScript } from './run-script.js';

// Each test gets its own script path so the fixed orchestrator location
// (`/tmp/task-script.sh`) is never touched and tests don't collide.
const tmpPaths: string[] = [];
function scriptPath(name: string): string {
  const p = path.join(os.tmpdir(), `run-script-test-${name}.sh`);
  tmpPaths.push(p);
  return p;
}

afterEach(() => {
  for (const p of tmpPaths.splice(0)) {
    if (fs.existsSync(p)) {
      fs.rmSync(p);
    }
  }
});

describe('runScript success + parse paths', () => {
  it('resolves ok with the parsed result for a well-formed precheck', async () => {
    const out = await runScript(
      'echo \'{"wake_agent": false, "data": {"reason": "quiet"}}\'',
      { scriptPath: scriptPath('ok-false') },
    );
    expect(out.ok).toBe(true);
    if (out.ok) {
      expect(out.result.wake_agent).toBe(false);
      expect(out.result.data).toEqual({ reason: 'quiet' });
    }
  });

  it('reads the trailing JSON line, ignoring earlier diagnostic output', async () => {
    const out = await runScript(
      'echo "warming up"; echo \'{"wake_agent": true, "data": {}}\'',
      { scriptPath: scriptPath('trailing-line') },
    );
    expect(out.ok).toBe(true);
    if (out.ok) {
      expect(out.result.wake_agent).toBe(true);
    }
  });

  it('flags empty stdout as empty-output', async () => {
    const out = await runScript('true', { scriptPath: scriptPath('empty') });
    expect(out).toEqual({ ok: false, reason: 'empty-output' });
  });

  it('flags non-JSON stdout as invalid-json', async () => {
    const out = await runScript('echo not json', {
      scriptPath: scriptPath('nonjson'),
    });
    expect(out.ok).toBe(false);
    if (!out.ok) {
      expect(out.reason).toBe('invalid-json');
    }
  });
});

describe('runScript execfile-error detail (#812 Bug A)', () => {
  it('carries the exit code but never the child stderr (no-secrets)', async () => {
    const out = await runScript('echo "SECRET_TOKEN=abc123" >&2; exit 3', {
      scriptPath: scriptPath('exit3'),
    });
    expect(out.ok).toBe(false);
    if (!out.ok) {
      expect(out.reason).toBe('execfile-error');
      expect(out.detail).toContain('exit=3');
      // stderr (which can carry secrets) must not reach the persisted detail.
      expect(out.detail).not.toContain('SECRET_TOKEN');
    }
  });

  it('resolves an execfile-error instead of throwing when the script cannot be written', async () => {
    // A write into a non-existent directory throws ENOENT synchronously;
    // runScript must catch it and resolve, not reject (the caller relies
    // on a persisted precheck-error row and does not try/catch).
    const out = await runScript('true', {
      scriptPath: path.join(
        os.tmpdir(),
        'run-script-test-no-such-dir',
        'task.sh',
      ),
    });
    expect(out.ok).toBe(false);
    if (!out.ok) {
      expect(out.reason).toBe('execfile-error');
      expect(out.detail).toContain('spawn-failed');
    }
  });
});

describe('runScript timeout escalation (#812 Bug B)', () => {
  it('SIGKILLs a SIGTERM-ignoring child after the grace window and reports the timeout', async () => {
    // `trap '' TERM` makes the child ignore SIGTERM; without SIGKILL
    // escalation this sleep would run 30s and the test would hang.
    const out = await runScript("trap '' TERM; sleep 30", {
      scriptPath: scriptPath('ignore-term'),
      timeoutMs: 150,
      killGraceMs: 100,
    });
    expect(out.ok).toBe(false);
    if (!out.ok) {
      expect(out.reason).toBe('execfile-error');
      expect(out.detail).toContain('timed out after 0s');
      expect(out.detail).toContain('signal=SIGKILL');
    }
  });

  it('stamps timedOut on a timeout kill and omits it otherwise (#890 follow-up)', async () => {
    // The operator alert keys off this structured field rather than
    // `detail`'s prose, so it must be set exactly on the timeout path.
    // The prose assertions above stay green even if the field is lost,
    // which is the regression this covers.
    const killed = await runScript('sleep 30', {
      scriptPath: scriptPath('timedout-flag'),
      timeoutMs: 120,
      killGraceMs: 100,
    });
    expect(killed.ok).toBe(false);
    if (!killed.ok) {
      expect(killed.timedOut).toBe(true);
    }

    // A non-zero exit is a failure but NOT a timeout — it must not
    // alert; it stays covered by the heartbeat's task-failure report.
    const crashed = await runScript('exit 3', {
      scriptPath: scriptPath('nonzero-exit'),
      timeoutMs: 10_000,
    });
    expect(crashed.ok).toBe(false);
    if (!crashed.ok) {
      expect(crashed.timedOut).toBeUndefined();
    }

    // Malformed output is likewise not a timeout.
    const badJson = await runScript('echo "not json"', {
      scriptPath: scriptPath('bad-json'),
      timeoutMs: 10_000,
    });
    expect(badJson.ok).toBe(false);
    if (!badJson.ok) {
      expect(badJson.timedOut).toBeUndefined();
    }
  });

  it('arms no timer when no budget is declared, so the container bounds the run (#890)', async () => {
    // With `timeoutMs` omitted the precheck runs to completion rather
    // than being killed by any in-runner default — the flat 30s global
    // is gone and the container kill is the only other bound.
    const out = await runScript(
      'sleep 0.3; echo \'{"wake_agent": false, "data": {}}\'',
      { scriptPath: scriptPath('no-declared-budget') },
    );
    expect(out.ok).toBe(true);
    if (out.ok) {
      expect(out.result.wake_agent).toBe(false);
    }
  });

  it('kills a grandchild in the process group, not just the bash parent', async () => {
    // bash backgrounds a SIGTERM-ignoring grandchild and waits on it. The
    // group SIGKILL must reach the grandchild, or `wait` never returns and
    // the run outlives the timeout+grace budget.
    const start = Date.now();
    const out = await runScript(
      "trap '' TERM; sleep 30 & child=$!; wait $child",
      {
        scriptPath: scriptPath('grandchild'),
        timeoutMs: 150,
        killGraceMs: 100,
      },
    );
    const elapsed = Date.now() - start;
    expect(out.ok).toBe(false);
    // Comfortably under the child's 30s sleep — proves the group was killed.
    expect(elapsed).toBeLessThan(5_000);
  });
});
