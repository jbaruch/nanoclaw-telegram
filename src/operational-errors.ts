const FILE_SYSTEM_ERROR_CODES = new Set([
  'EACCES',
  'EBUSY',
  'EEXIST',
  'EIO',
  'EISDIR',
  'ELOOP',
  'EMFILE',
  'ENAMETOOLONG',
  'ENFILE',
  'ENOENT',
  'ENOSPC',
  'ENOTDIR',
  'ENOTEMPTY',
  'EPERM',
  'EROFS',
  'ESTALE',
]);

const NETWORK_ERROR_CODES = new Set([
  'EAI_AGAIN',
  'ECONNABORTED',
  'ECONNREFUSED',
  'ECONNRESET',
  'EHOSTUNREACH',
  'ENETDOWN',
  'ENETUNREACH',
  'ENOTFOUND',
  'EPIPE',
  'ETIMEDOUT',
]);

const SPAWN_ERROR_CODES = new Set([
  ...FILE_SYSTEM_ERROR_CODES,
  'E2BIG',
  'EAGAIN',
  'ENOMEM',
  'ETXTBSY',
]);

function errorCode(err: unknown): string | undefined {
  if (!(err instanceof Error) || !('code' in err)) return undefined;
  const code = (err as NodeJS.ErrnoException).code;
  return typeof code === 'string' ? code : undefined;
}

export function hasOperationalErrorCode(
  err: unknown,
  ...codes: string[]
): err is NodeJS.ErrnoException {
  const code = errorCode(err);
  return code !== undefined && codes.includes(code);
}

export function isFileSystemError(err: unknown): err is NodeJS.ErrnoException {
  const code = errorCode(err);
  return code !== undefined && FILE_SYSTEM_ERROR_CODES.has(code);
}

export function isNetworkError(err: unknown): err is NodeJS.ErrnoException {
  const code = errorCode(err);
  return code !== undefined && NETWORK_ERROR_CODES.has(code);
}

export function isSpawnError(err: unknown): err is NodeJS.ErrnoException {
  const code = errorCode(err);
  return code !== undefined && SPAWN_ERROR_CODES.has(code);
}

interface ExecFailure extends Error {
  status?: number | null;
  signal?: string | null;
  stdout?: unknown;
  stderr?: unknown;
}

export function isExecFailure(err: unknown): err is ExecFailure {
  if (isSpawnError(err)) return true;
  if (!(err instanceof Error) || err instanceof TypeError) return false;

  const failure = err as ExecFailure;
  const exited = typeof failure.status === 'number';
  const signalled =
    failure.status === null && typeof failure.signal === 'string';
  return (exited || signalled) && 'stdout' in failure && 'stderr' in failure;
}

interface MessageServiceError {
  code?: unknown;
  error_code?: unknown;
  error?: { code?: unknown };
}

function serviceStatus(err: unknown): number | undefined {
  if (typeof err !== 'object' || err === null) return undefined;
  const serviceError = err as MessageServiceError;
  if (typeof serviceError.error_code === 'number') {
    return serviceError.error_code;
  }
  if (typeof serviceError.code === 'number') return serviceError.code;
  return undefined;
}

export function isMessageServiceError(err: unknown): err is Error {
  const status = serviceStatus(err);
  return (
    err instanceof Error &&
    !(err instanceof TypeError) &&
    (isNetworkError(err) ||
      (status !== undefined && status >= 400 && status <= 599))
  );
}

export function isTransientMessageServiceError(err: unknown): boolean {
  if (!isMessageServiceError(err)) return false;
  const status = serviceStatus(err);
  return isNetworkError(err) || (status !== undefined && status >= 500);
}

export class MessageDeliveryError extends Error {
  constructor(cause: Error) {
    super(`Message delivery failed: ${cause.message}`, { cause });
    this.name = 'MessageDeliveryError';
  }
}
