// Scheduled-task precheck runner, extracted from `index.ts` (#812) so the
// timeout / process-group / error-detail behaviour is unit-testable without
// importing the runner's top-level `main()`.
//
// The precheck script is the OUTER PROCESS BOUNDARY of the scheduled-task
// contract: the scheduler reads a non-zero exit OR malformed stdout as
// "skip waking the agent this cycle". `runScript` never throws — every
// failure mode resolves to a discriminated `RunScriptResult` so the caller
// can persist a specific `task_run_logs.error` via `buildPrecheckErrorOutput`.

import { spawn } from 'child_process';
import * as fs from 'fs';

import {
  formatExecErrorDetail,
  type PrecheckErrorReason,
} from './precheck-emission.js';
import { parseScriptOutput, type ScriptResult } from './script-output-parse.js';

export const SCRIPT_TIMEOUT_MS = 30_000;
// #812 Bug B: after the timeout fires we SIGTERM the child's process
// GROUP; a precheck blocked in a network syscall (the Google gateway,
// `gh`) ignores SIGTERM and would otherwise run to its own socket
// timeout (110s observed vs the nominal 30s), holding the maintenance
// slot. We hard-SIGKILL the whole group after this grace window.
// `detached` (below) puts the child in its own group so a python
// grandchild forked by bash is killed too, not just bash.
export const SCRIPT_KILL_GRACE_MS = 5_000;
const SCRIPT_MAX_OUTPUT_CHARS = 1024 * 1024;

// Discriminated return shape (#581 follow-up): on success carries the
// parsed `ScriptResult`; on failure carries a `PrecheckErrorReason` so
// `buildPrecheckErrorOutput` can name what actually went wrong, plus an
// optional free-form `detail` (#812 Bug A) that disambiguates an
// `execfile-error` — timeout-kill vs signal vs non-zero exit vs spawn
// failure — which the bare reason string could not.
export type RunScriptResult =
  | { ok: true; result: ScriptResult }
  | { ok: false; reason: PrecheckErrorReason; detail?: string };

export interface RunScriptOptions {
  timeoutMs?: number;
  killGraceMs?: number;
  // Injected so the caller (index.ts) keeps its `[agent-runner]`-prefixed
  // container-log format; defaults to a no-op for tests.
  log?: (message: string) => void;
  // Overridable so tests write to a tmp path instead of the fixed
  // orchestrator location.
  scriptPath?: string;
}

export function runScript(
  script: string,
  opts: RunScriptOptions = {},
): Promise<RunScriptResult> {
  const timeoutMs = opts.timeoutMs ?? SCRIPT_TIMEOUT_MS;
  const killGraceMs = opts.killGraceMs ?? SCRIPT_KILL_GRACE_MS;
  const log = opts.log ?? ((): void => {});
  const scriptPath = opts.scriptPath ?? '/tmp/task-script.sh';

  return new Promise((resolve) => {
    // Write + spawn inside the executor so a synchronous failure (write
    // permission / disk full, or spawn arg error) resolves an
    // execfile-error rather than throwing out of `runScript` — the caller
    // relies on a persisted precheck-error row and does not catch (#812).
    let child: ReturnType<typeof spawn>;
    try {
      fs.writeFileSync(scriptPath, script, { mode: 0o755 });
      // `detached` puts the child in its own process group so a python
      // grandchild forked by bash is killed with the group, not orphaned
      // when we only signal bash (#812 Bug B).
      //
      // stderr is discarded at the OS level (`'ignore'`): child stderr is
      // untrusted script output that may carry secrets, so it must never
      // reach any log or the persisted error (`coding-policy: no-secrets`).
      // Diagnosis relies on the controlled exit/signal fields instead.
      // Discarding also avoids a full stderr pipe blocking the child.
      child = spawn('bash', [scriptPath], {
        env: process.env,
        detached: true,
        stdio: ['ignore', 'pipe', 'ignore'],
      });
      // outer-boundary-process-contract (coding-policy: error-handling):
      // runScript is the precheck process boundary and promises never to
      // throw — the caller consumes its RunScriptResult and does not
      // try/catch (#812).
      //   - Caller's silent-failure shape: index.ts awaits runScript and
      //     persists task_run_logs.error from the result; a throw here
      //     would skip buildPrecheckErrorOutput's specific precheck row.
      //   - What the catch emits: resolves an `execfile-error` result so
      //     the failure lands as a diagnosable precheck-error row.
      //   - Why propagation breaks the contract: a rejected promise would
      //     leave the setup failure without a precheck-shaped error.
      // eslint-disable-next-line no-catch-all/no-catch-all -- outer-boundary-process-contract
    } catch (err) {
      const detail = formatExecErrorDetail({
        spawnCode: (err as NodeJS.ErrnoException).code ?? 'setup-error',
      });
      log(`Script error: ${detail}`);
      resolve({ ok: false, reason: 'execfile-error', detail });
      return;
    }

    let stdout = '';
    let timedOut = false;
    let spawnCode: string | undefined;
    let settled = false;
    let escalationTimer: NodeJS.Timeout | undefined;

    // Signal the whole process group (negative pid). ESRCH means the
    // group already exited between our decision to kill and the call —
    // benign. Any other errno is surfaced to the container log but must
    // NOT throw: `killGroup` runs inside a timer callback, where a throw
    // would crash the runner and prevent ANY precheck-error row from
    // being written (#812).
    const killGroup = (signal: NodeJS.Signals): void => {
      if (child.pid === undefined) {
        return;
      }
      try {
        process.kill(-child.pid, signal);
        // outer-boundary-process-contract (coding-policy: error-handling):
        // killGroup runs inside a setTimeout callback on the precheck
        // boundary.
        //   - Caller's silent-failure shape: index.ts persists
        //     task_run_logs.error from runScript's resolved result.
        //   - What the catch emits: ESRCH (group already exited) is
        //     benign and dropped; any other errno is logged to the
        //     container log with context.
        //   - Why propagation breaks the contract: a throw from this
        //     timer callback is an uncaught exception that crashes the
        //     runner, so no precheck-error row is ever written (#812).
        // eslint-disable-next-line no-catch-all/no-catch-all -- outer-boundary-process-contract
      } catch (err) {
        const code = (err as NodeJS.ErrnoException).code;
        if (code !== 'ESRCH') {
          log(`Failed to ${signal} script process group: ${code ?? err}`);
        }
      }
    };

    const timeoutTimer = setTimeout(() => {
      timedOut = true;
      killGroup('SIGTERM');
      escalationTimer = setTimeout(() => killGroup('SIGKILL'), killGraceMs);
    }, timeoutMs);

    // Settle once, from whichever of `error` / `close` fires first. A
    // failed spawn (ENOENT) emits `error` and may never emit `close`, so
    // both paths must be able to resolve.
    const finish = (
      code: number | null,
      signal: NodeJS.Signals | null,
    ): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeoutTimer);
      if (escalationTimer) {
        clearTimeout(escalationTimer);
      }

      if (spawnCode || timedOut || code !== 0 || signal !== null) {
        const detail = formatExecErrorDetail({
          exitCode: code,
          signal,
          timedOut,
          timeoutMs,
          spawnCode,
        });
        log(`Script error: ${detail}`);
        resolve({ ok: false, reason: 'execfile-error', detail });
        return;
      }

      const outcome = parseScriptOutput(stdout);
      if (!outcome.ok) {
        const lastLine = (outcome.lastLine ?? '').slice(0, 200);
        if (outcome.reason === 'empty') {
          log('Script produced no output');
          resolve({ ok: false, reason: 'empty-output' });
        } else if (outcome.reason === 'invalid_json') {
          log(`Script output is not valid JSON: ${lastLine}`);
          resolve({ ok: false, reason: 'invalid-json' });
        } else if (outcome.reason === 'invalid_data_shape') {
          log(
            `Script output 'data' must be a JSON object per coding-policy: script-delegation: ${lastLine}`,
          );
          resolve({ ok: false, reason: 'invalid-data-shape' });
        } else {
          log(`Script output missing wake_agent boolean: ${lastLine}`);
          resolve({ ok: false, reason: 'missing-wake-agent' });
        }
        return;
      }
      resolve({ ok: true, result: outcome.result });
    };

    // Keep the tail on overflow — prechecks emit a single trailing JSON
    // line, so `parseScriptOutput` reads the end, not the head.
    child.stdout?.on('data', (chunk: Buffer) => {
      const next = stdout + chunk.toString();
      stdout =
        next.length > SCRIPT_MAX_OUTPUT_CHARS
          ? next.slice(-SCRIPT_MAX_OUTPUT_CHARS)
          : next;
    });

    // Fires when bash itself can't be spawned (e.g. ENOENT). `close` may
    // not follow, so settle here too.
    child.on('error', (err) => {
      spawnCode = (err as NodeJS.ErrnoException).code ?? 'spawn-error';
      finish(null, null);
    });
    child.on('close', (code, signal) => {
      finish(code, signal);
    });
  });
}
