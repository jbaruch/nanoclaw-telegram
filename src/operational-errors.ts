import { GrammyError, HttpError } from 'grammy';

/** Node filesystem failures that callers can report or explicitly recover from. */
export function isFileSystemError(
  error: unknown,
): error is NodeJS.ErrnoException {
  return (
    error instanceof Error &&
    'code' in error &&
    typeof error.code === 'string' &&
    [
      'ENOENT',
      'EACCES',
      'EPERM',
      'EIO',
      'ENOSPC',
      'EMFILE',
      'ENFILE',
      'EROFS',
      'ENOTDIR',
      'EISDIR',
      'ELOOP',
      'EEXIST',
      'EBUSY',
      'EXDEV',
      'ENAMETOOLONG',
      'ENOTEMPTY',
    ].includes(error.code)
  );
}

export function hasErrno(
  error: unknown,
  code: string,
): error is NodeJS.ErrnoException {
  return error instanceof Error && 'code' in error && error.code === code;
}

/** execSync's failed-child result, distinct from programming/argument errors. */
export function isCommandError(error: unknown): error is Error {
  return (
    error instanceof Error &&
    (('status' in error &&
      (typeof error.status === 'number' || error.status === null) &&
      'signal' in error &&
      (typeof error.signal === 'string' || error.signal === null)) ||
      hasErrno(error, 'ENOENT') ||
      hasErrno(error, 'EACCES'))
  );
}

export function isTelegramError(
  error: unknown,
): error is GrammyError | HttpError {
  return error instanceof GrammyError || error instanceof HttpError;
}

/** cron-parser 5.5 exposes input validation as plain Error, without an error subtype. */
export function isCronInputError(error: unknown): error is Error {
  return (
    error instanceof Error &&
    error.constructor === Error &&
    /^(Invalid (cron expression|characters,|range:|step:|list value format|repeat:|explicit day of month definition|last weekday of the month expression:)|Validation error,|Constraint error,|Cron(?:Second|Minute|Hour|DayOfMonth|Month|DayOfWeek) Validation error,|Out of the time span range|Invalid time value)/.test(
      error.message,
    )
  );
}

/** Errors emitted by Node's HTTPS client while fetching media. */
export function isNetworkError(error: unknown): error is NodeJS.ErrnoException {
  return (
    error instanceof Error &&
    'code' in error &&
    typeof error.code === 'string' &&
    [
      'ENOTFOUND',
      'ECONNRESET',
      'ETIMEDOUT',
      'ECONNREFUSED',
      'EAI_AGAIN',
      'EPIPE',
      'ECONNABORTED',
      'ENETUNREACH',
      'EHOSTUNREACH',
      'ERR_TLS_CERT_ALTNAME_INVALID',
      'CERT_HAS_EXPIRED',
      'DEPTH_ZERO_SELF_SIGNED_CERT',
      'UNABLE_TO_VERIFY_LEAF_SIGNATURE',
    ].includes(error.code)
  );
}
