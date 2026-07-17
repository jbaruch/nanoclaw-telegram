import { describe, it, expect } from 'vitest';

import {
  ContainerAgentError,
  toContainerAgentError,
} from './container-runner.js';

/**
 * #784 — runContainerAgent raises a typed ContainerAgentError for its
 * infrastructure failures so runAgent can narrow to a closed set instead of a
 * defect-blacklist. These exercise the classifier `toContainerAgentError`
 * directly: the infra/defect line is the module's #735 no-catch-all
 * convention.
 */

function errno(code: string, message = `${code}: simulated`): Error {
  const err = new Error(message);
  (err as NodeJS.ErrnoException).code = code;
  return err;
}

describe('toContainerAgentError', () => {
  describe('converts operational failures to ContainerAgentError', () => {
    it('wraps an fs ErrnoException with an operational code, preserving cause', () => {
      // Guards that the catch actually ran (toContainerAgentError is typed
      // `never`, so the try can't fall through).
      expect.assertions(2);
      const cause = errno('EACCES', 'EACCES: permission denied, mkdir');
      try {
        toContainerAgentError(cause);
      } catch (err) {
        // Rethrow anything unexpected; assert on the typed error we expect.
        if (!(err instanceof ContainerAgentError)) throw err;
        expect(err.message).toBe(cause.message);
        expect(err.cause).toBe(cause);
      }
    });

    it('wraps the other spawn-path fs codes', () => {
      for (const code of ['EROFS', 'ENOSPC', 'EEXIST', 'EBUSY']) {
        expect(() => toContainerAgentError(errno(code))).toThrow(
          ContainerAgentError,
        );
      }
    });
  });

  describe('propagates programmer defects untouched', () => {
    it('re-raises a code-less plain Error (validation / init-guard defect)', () => {
      // e.g. rebuildCadenceRegistryForGroup's `called before initDatabase`
      // guard, or resolveGroupFolderPath's escaping-folder check. These are
      // defects that must surface, NOT operational failures — the classifier
      // only converts fs ErrnoExceptions. Operational non-fs failures (OneCLI
      // fail-closed) are typed at their sites, never through this path.
      const cause = new Error(
        'rebuildCadenceRegistry called before initDatabase',
      );
      expect(() => toContainerAgentError(cause)).toThrow(cause);
      expect(() => toContainerAgentError(cause)).not.toThrow(
        ContainerAgentError,
      );
    });

    it('re-raises a TypeError, not a ContainerAgentError', () => {
      const cause = new TypeError('Cannot read properties of undefined');
      expect(() => toContainerAgentError(cause)).toThrow(cause);
      expect(() => toContainerAgentError(cause)).not.toThrow(
        ContainerAgentError,
      );
    });

    it('re-raises ReferenceError / RangeError / SyntaxError', () => {
      for (const cause of [
        new ReferenceError('x is not defined'),
        new RangeError('out of range'),
        new SyntaxError('bad json'),
      ]) {
        expect(() => toContainerAgentError(cause)).toThrow(cause);
        expect(() => toContainerAgentError(cause)).not.toThrow(
          ContainerAgentError,
        );
      }
    });

    it('re-raises an fs ErrnoException with an UNEXPECTED code (the guard-rethrow signal)', () => {
      // EIO is not in CR_FS_CODES — the same code shape the module's
      // isFsErrorWithCode guards rethrow as a defect. It must NOT become a
      // routine operational error.
      const cause = errno('EIO', 'EIO: i/o error');
      expect(() => toContainerAgentError(cause)).toThrow(cause);
      expect(() => toContainerAgentError(cause)).not.toThrow(
        ContainerAgentError,
      );
    });

    it('re-raises a non-Error throw as itself', () => {
      expect(() => toContainerAgentError('not an error object')).toThrow(
        'not an error object',
      );
    });
  });

  it('passes an already-typed ContainerAgentError through unchanged', () => {
    expect.assertions(2);
    const original = new ContainerAgentError('already typed');
    try {
      toContainerAgentError(original);
    } catch (err) {
      if (!(err instanceof ContainerAgentError)) throw err;
      // Same instance re-raised, not re-wrapped.
      expect(err).toBe(original);
      expect(err.cause).toBeUndefined();
    }
  });
});

describe('ContainerAgentError', () => {
  it('is an Error subclass with a stable name and optional cause', () => {
    const cause = new Error('root');
    const err = new ContainerAgentError('boom', { cause });
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(ContainerAgentError);
    expect(err.name).toBe('ContainerAgentError');
    expect(err.message).toBe('boom');
    expect(err.cause).toBe(cause);
  });
});
