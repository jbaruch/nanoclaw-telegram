/**
 * Shared filesystem-error policy for best-effort writers and sweepers.
 *
 * Background: parts of the orchestrator do best-effort filesystem work
 * (writing IPC sentinels, sweeping stale inputs, etc.) and must NOT
 * block the message-delivery path on every transient errno. The policy
 * here is "tolerate codes that legitimately occur on the IPC dirs;
 * propagate everything else so a real bug (TypeError, ReferenceError,
 * unrelated programming error) surfaces instead of being silently
 * swallowed."
 *
 * Originally inlined in `src/group-queue.ts`; extracted here so writers
 * (group-queue) and sweepers (ipc-input-sweep) can't drift on which
 * errnos they treat as expected. If a new caller needs the same
 * policy, import from here rather than redefining the set.
 */
const EXPECTED_FS_ERROR_CODES = new Set([
  'EACCES',
  'EPERM',
  'ENOSPC',
  'EROFS',
  'ENOENT',
  'EISDIR',
  'EBUSY',
]);

/**
 * Returns true when `err` is an Error with a `.code` matching the
 * shared expected-errno set. Anything else (including non-Error values
 * thrown directly, e.g. strings or programming bugs) returns false so
 * the caller's `if (!isExpectedFsError(err)) throw err;` path fires.
 */
export function isExpectedFsError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const code = (err as NodeJS.ErrnoException).code;
  return typeof code === 'string' && EXPECTED_FS_ERROR_CODES.has(code);
}

/**
 * Returns true when `err` is an `Error` carrying an errno `.code` that is
 * one of `codes`. Unlike `isExpectedFsError` (which pins the shared
 * best-effort set), this lets a caller state the exact codes it treats as
 * recoverable for a specific syscall — e.g. `['ESRCH', 'EPERM']` for a
 * `process.kill(pid, 0)` liveness probe. Any non-Error value, or a
 * different (or absent) code, returns false so the caller's
 * `if (!isFsErrorWithCode(err, [...])) throw err;` path re-throws it.
 */
export function isFsErrorWithCode(
  err: unknown,
  codes: readonly string[],
): boolean {
  if (!(err instanceof Error)) return false;
  const code = (err as NodeJS.ErrnoException).code;
  return typeof code === 'string' && codes.includes(code);
}

/**
 * Returns true when `err` is any errno-coded system Error (an `Error` with a
 * string `.code`), regardless of the specific code. Use this ONLY on
 * cleanup-after-error and best-effort teardown paths, where the goal is to
 * never let a secondary cleanup failure (EIO, EDQUOT, ENOTEMPTY, or any other
 * fs/OS errno) mask the primary in-flight error. Genuine defects
 * (`TypeError`, `ReferenceError` — no `.code`) still return false so the
 * caller's `if (!isErrnoCodedError(err)) throw err;` path re-throws them.
 *
 * Primary best-effort fs operations (not masking another error) should keep
 * using `isFsErrorWithCode` with an explicit per-syscall set so an unexpected
 * errno surfaces.
 */
export function isErrnoCodedError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  return typeof (err as NodeJS.ErrnoException).code === 'string';
}
