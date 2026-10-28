/** Expected native filesystem failures; implementation errors remain observable. */
export function hasErrno(
  error: unknown,
  code: string,
): error is NodeJS.ErrnoException {
  return error instanceof Error && 'code' in error && error.code === code;
}
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
