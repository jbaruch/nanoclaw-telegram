/**
 * Guard for child_process failures (execSync / execFileSync).
 *
 * Best-effort capability probes ("is docker installed?", "is the service
 * running?") must swallow only a genuine subprocess failure — the command
 * exited non-zero, was killed / timed out, or failed to spawn — and let an
 * unexpected defect (a TypeError from a bad call, a non-Error throw)
 * propagate, per `coding-policy: error-handling` ("catch specific exception
 * types, never bare catch-all handlers").
 */

/**
 * Errno-style codes a `child_process` spawn failure carries in `.code`.
 * Deliberately a closed set: Node *programming* errors also use string
 * codes (`ERR_INVALID_ARG_TYPE`, `ERR_INVALID_ARG_VALUE`, …), so matching
 * "any string code" would swallow real defects. Only genuine spawn errnos
 * count; everything else propagates.
 */
const SPAWN_ERROR_CODES = new Set([
  'ENOENT', // binary not found on PATH
  'EACCES', // found but not executable
  'EPERM', // not permitted to spawn
  'E2BIG', // argument list too long
  'ENOMEM', // out of memory spawning
  'EMFILE', // too many open files
  'ENFILE',
  'ETIMEDOUT', // spawn timed out
]);

/**
 * Returns true when `err` is a genuine `child_process` failure: the command
 * exited non-zero or was killed/timed out (`status` / `signal` present, even
 * if null), or it failed to spawn (an errno-style `code` in the closed set
 * above). A Node programming error (a bad-argument `ERR_*`, a plain
 * `Error`, a non-Error throw) carries none of these, so the caller's
 * `if (!isSubprocessError(err)) throw err;` path re-throws it.
 */
export function isSubprocessError(err: unknown): boolean {
  if (!(err instanceof Error)) return false;
  const e = err as NodeJS.ErrnoException & {
    status?: number | null;
    signal?: NodeJS.Signals | null;
  };
  if ('status' in e || 'signal' in e) return true;
  return typeof e.code === 'string' && SPAWN_ERROR_CODES.has(e.code);
}
