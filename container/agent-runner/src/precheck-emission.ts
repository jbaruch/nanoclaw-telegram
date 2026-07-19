// #581 follow-up — split the agent-runner's precheck-phase exit into
// two distinct shapes so silent-success watchdogs can act on the row
// in `task_run_logs`.
//
// History: PR #591 fixed the happy-path silence emit so wrapper
// scheduled tasks running to completion landed a non-null
// `task_run_logs.result`. The verification window after merge
// (`jbaruch/nanoclaw#581` comments dated 2026-05-20 onwards) found a
// distinct shape AyeAye kept flagging: fires that lasted 4–6 s with
// `status='success'` + `result=null`. Tracing those run windows
// against the session JSONL showed the agent never woke for any of
// them — the agent-runner exited inside the `runScript` precheck
// branch at `index.ts:4421-4435` with the legacy
// `writeOutput({status: 'success', result: null})` path.
//
// That single branch conflated two distinct outcomes that should
// land in `task_run_logs` differently:
//
//   - Healthy `wake_agent: false` (precheck succeeded, decided no
//     work needed — e.g. cadence cap not yet elapsed). Today's
//     `nightly-external-sync` cursor at 2026-05-20T04:06:58Z + a
//     3-day cap means every fire for the next 72 h SHOULD short-
//     circuit; the watchdog should be able to recognise the row as
//     healthy quiet.
//
//   - Genuine precheck failure (script crashed, emitted non-JSON,
//     omitted `wake_agent`). Pre-fix this produced the same shape as
//     the healthy case; the gating decision was unknown but the row
//     said `success`. The watchdog cannot act on an unknown gate.
//
// Fix shape: a dedicated `'precheck_skipped'` status (added to
// `ContainerOutput` and `TaskRunLog`) marks the healthy case with a
// non-null diagnostic result that embeds the precheck's `data`
// payload verbatim inside an `<internal>` envelope. The failure case
// becomes `status: 'error'` with a diagnostic result + `error`
// message — surfaced through the existing error pipeline so the next
// run isn't masked by an empty success row.
//
// Pulled out as a pure helper following the precedent of
// `result-suppression.ts` (PR #591), `silent-stop-synthesis.ts`, and
// `format-error-result.ts`: the agent-runner's emission decision is
// unit-testable without spinning the SDK iterator or the
// `runScript` execFile.

export interface PrecheckSkippedOutput {
  status: 'precheck_skipped';
  result: string;
}

export interface PrecheckErrorOutput {
  status: 'error';
  result: string;
  error: string;
}

/**
 * Build the writeOutput payload when the precheck script returned
 * `wake_agent: false`. The precheck's `data` payload (the gating
 * decision's diagnostic fields — `reason`, `last_run`, `age_hours`,
 * `cadence_hours`, etc. for cadence-cap prechecks) is JSON-encoded
 * verbatim inside an `<internal>` envelope so operator audits can
 * read what the precheck saw. `<internal>` tags are stripped by the
 * orchestrator and task-scheduler before user-visible chat-echo (see
 * the `cleanResult` step in `src/task-scheduler.ts:1119`), so the
 * envelope lands in `task_run_logs.result` without producing chat
 * noise.
 *
 * Per `coding-policy: script-delegation`, prechecks emit
 * `{"wake_agent": false, "data": {}}` (object-shaped `data`); the
 * parser at `script-output-parse.ts` accepts an omitted `data` as
 * valid (defaults to `{}` semantically). `data === undefined` would
 * make `JSON.stringify(data)` emit the literal `undefined` (not
 * valid JSON), so the helper coerces to `{}` before serialising.
 */
export function buildPrecheckSkippedOutput(
  data: unknown,
): PrecheckSkippedOutput {
  const payload = data === undefined ? {} : data;
  return {
    status: 'precheck_skipped',
    result: `<internal>precheck-skipped: ${JSON.stringify(payload)}</internal>`,
  };
}

/**
 * Build the writeOutput payload when the precheck script failed.
 * `runScript` returns `null` for multiple failure modes — `execFile`
 * error (crash / timeout / nonzero exit), empty stdout, non-JSON
 * stdout, invalid `data` shape, or missing `wake_agent`. The caller
 * passes the specific reason through so the persisted diagnostic
 * tells the operator what actually went wrong rather than a generic
 * "something failed" string. The gating decision is unknown — the
 * row lands as `'error'` (flows through the existing error pipeline)
 * rather than `'success'` which would mask the failure as a healthy
 * row to the silent-success watchdog.
 */
export type PrecheckErrorReason =
  | 'execfile-error'
  | 'empty-output'
  | 'invalid-json'
  | 'invalid-data-shape'
  | 'missing-wake-agent';

export function buildPrecheckErrorOutput(
  reason: PrecheckErrorReason,
  detail?: string,
): PrecheckErrorOutput {
  const suffix = detail ? ` (${detail})` : '';
  const errorMsg = `precheck script failed: ${reason}${suffix}`;
  return {
    status: 'error',
    result: `<internal>precheck-error: ${reason}${suffix}</internal>`,
    error: errorMsg,
  };
}

// #812 Bug A: format the disambiguating cause of an `execfile-error` into
// a single line for `task_run_logs.error`. Before this, every execFile
// failure — timeout-kill, signal, genuine non-zero exit, spawn failure —
// collapsed to the bare reason `execfile-error`, so a hung precheck killed
// at the timeout was indistinguishable from a script that exited 1. The
// runtime child fields (`killed`/`signal`/`code`) are exactly what
// disambiguates them.
//
// The child's stderr is deliberately NOT included here: this string is
// persisted to `task_run_logs.error` AND surfaced in the operator's chat
// alert, so raw stderr could leak secret-bearing diagnostics into durable
// logs (`coding-policy: no-secrets`). `runScript` discards child stderr
// entirely (spawned with stderr `'ignore'`), so diagnosis rests on these
// controlled fields. Excluding untrusted text also keeps the output a
// controlled string (signal names, our literals, integers) that is always
// safe to embed in the `<internal>` envelope. Pure so it is unit-tested
// directly; `runScript` threads its output into `buildPrecheckErrorOutput`.
export interface ExecErrorInfo {
  exitCode?: number | null;
  signal?: NodeJS.Signals | null;
  timedOut?: boolean;
  timeoutMs?: number;
  spawnCode?: string; // e.g. 'ENOENT' when `bash` itself can't be spawned
}

export function formatExecErrorDetail(info: ExecErrorInfo): string {
  const { exitCode, signal, timedOut, timeoutMs, spawnCode } = info;
  if (spawnCode) {
    return `spawn-failed: ${spawnCode}`;
  }
  const parts: string[] = [];
  if (timedOut) {
    parts.push(`timed out after ${Math.round((timeoutMs ?? 0) / 1000)}s`);
  }
  if (signal) {
    parts.push(`signal=${signal}`);
  }
  if (typeof exitCode === 'number' && exitCode !== 0) {
    parts.push(`exit=${exitCode}`);
  }
  return parts.join(', ') || 'unknown-failure';
}
