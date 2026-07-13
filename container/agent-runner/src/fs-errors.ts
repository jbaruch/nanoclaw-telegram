/**
 * Filesystem-error guards for best-effort agent-runner paths.
 *
 * Best-effort cleanup, sentinel-return, and probe catches must swallow
 * only the expected errno conditions and let anything unexpected (a
 * TypeError, a bug in the surrounding code, a non-Error throw) propagate
 * — per `coding-policy: error-handling` ("catch specific exception
 * types, never bare catch-all handlers"). The guard proves the invariant
 * once instead of scattering `instanceof`/`.code` checks.
 */

/**
 * Errno codes a best-effort filesystem operation may legitimately hit.
 * Mirrors `src/fs-errors.ts` in the orchestrator package — the two can't
 * share a module across the package boundary, so the set is duplicated
 * deliberately. Anything outside it (or a non-Error throw) is a real
 * defect and re-thrown by the caller.
 */
const EXPECTED_FS_ERROR_CODES = [
  'EACCES',
  'EPERM',
  'ENOSPC',
  'EROFS',
  'ENOENT',
  'EISDIR',
  'EBUSY',
] as const;

/**
 * Returns true when `err` is an `Error` carrying an errno `.code` that is
 * one of `codes`. Any non-Error value, or an Error with a different (or
 * absent) code, returns false so the caller's
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

/** True for an Error whose errno code is in the standard best-effort set. */
export function isExpectedFsError(err: unknown): boolean {
  return isFsErrorWithCode(err, EXPECTED_FS_ERROR_CODES);
}
