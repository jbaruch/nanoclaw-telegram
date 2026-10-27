import { GrammyError, HttpError } from 'grammy';
import Database from 'better-sqlite3';

/** Application-level delivery failed without a native Telegram acknowledgment. */
export class TelegramDeliveryError extends Error {}

/** A destination lacks a connected channel, operation or native target. */
export class IpcRouteUnavailableError extends Error {}

/** Native runtime exit, distinct from a callback implementation defect. */
export class ContainerExitError extends Error {
  constructor(readonly exitCode: number | null) {
    super(`Container exited with code ${exitCode}`);
  }
}

/** Recover only known operational causes; mixed/empty/cyclic aggregates remain defects. */
export function isOperationalRunError(error: unknown): boolean {
  let remaining = 128;
  const classify = (
    value: unknown,
    parents: Set<AggregateError>,
    depth: number,
  ): boolean => {
    if (--remaining < 0 || depth > 8) return false;
    if (value instanceof AggregateError) {
      if (
        parents.has(value) ||
        !Array.isArray(value.errors) ||
        value.errors.length === 0 ||
        value.errors.length > 32
      )
        return false;
      const next = new Set(parents).add(value);
      return (
        value.errors.every((item: unknown) =>
          classify(item, next, depth + 1),
        ) &&
        (!('cause' in value) || classify(value.cause, next, depth + 1))
      );
    }
    return (
      isFileSystemError(value) ||
      isCommandError(value) ||
      isTelegramError(value) ||
      hasErrno(value, 'EPIPE') ||
      hasErrno(value, 'ECONNRESET') ||
      value instanceof Database.SqliteError ||
      value instanceof IpcRouteUnavailableError ||
      value instanceof ContainerExitError
    );
  };
  return classify(error, new Set(), 0);
}

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
): error is GrammyError | HttpError | TelegramDeliveryError {
  return (
    error instanceof GrammyError ||
    error instanceof HttpError ||
    error instanceof TelegramDeliveryError
  );
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
