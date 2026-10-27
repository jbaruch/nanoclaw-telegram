import { execFileSync } from 'child_process';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { CronExpressionParser } from 'cron-parser';
import Database from 'better-sqlite3';
import { HttpError } from 'grammy';
import { describe, expect, it, vi } from 'vitest';
import {
  hasErrno,
  isFileSystemError,
  isCommandError,
  isTelegramError,
  isCronInputError,
  isNetworkError,
  isOperationalRunError,
  isRecoverableSqliteError,
  IpcRouteUnavailableError,
  ContainerExitError,
} from './operational-errors.js';

function caught(operation: () => unknown): unknown {
  const call = vi.fn(operation);
  expect(call).toThrow();
  return call.mock.results[0].value;
}

describe('expected operational errors', () => {
  it('recovers known pipe and all-operational aggregate failures, never mixed or cyclic defects', () => {
    const pipe = Object.assign(new Error('fixed pipe failure'), {
      code: 'EPIPE',
    });
    const reset = Object.assign(new Error('fixed reset'), {
      code: 'ECONNRESET',
    });
    const transport = new HttpError('fixed transport', new Error('synthetic'));
    const command = Object.assign(new Error('fixed native stop failure'), {
      status: 7,
      signal: null,
    });
    const combined = new AggregateError(
      [transport, command],
      'fixed combined failure',
      { cause: transport },
    );
    for (const error of [
      pipe,
      reset,
      combined,
      new IpcRouteUnavailableError('fixed unavailable route'),
      new ContainerExitError(1),
    ])
      expect(isOperationalRunError(error)).toBe(true);
    const defect = new TypeError('fixed implementation defect');
    for (const error of [
      new AggregateError([transport, defect]),
      new AggregateError([transport], 'fixed hidden cause', { cause: defect }),
      new AggregateError([]),
      new Error('unclassified'),
    ])
      expect(isOperationalRunError(error)).toBe(false);
    const cycle = new AggregateError([]);
    cycle.errors.push(cycle);
    expect(isOperationalRunError(cycle)).toBe(false);
  });

  it('propagates actual invalid SQL/schema errors while recognizing only enumerated contention codes', () => {
    const database = new Database(':memory:');
    try {
      const error = caught(() =>
        database.prepare('SELECT missing_column FROM missing_table'),
      );
      expect(error).toMatchObject({ code: 'SQLITE_ERROR' });
      expect(isRecoverableSqliteError(error)).toBe(false);
      expect(isOperationalRunError(error)).toBe(false);
      for (const code of ['SQLITE_BUSY', 'SQLITE_LOCKED', 'SQLITE_IOERR_READ'])
        expect(
          isRecoverableSqliteError(
            new Database.SqliteError('fixed storage condition', code),
          ),
        ).toBe(true);
      for (const code of [
        'SQLITE_ERROR',
        'SQLITE_SCHEMA',
        'SQLITE_MISUSE',
        'SQLITE_CONSTRAINT_UNIQUE',
        'SQLITE_UNKNOWN',
      ])
        expect(
          isRecoverableSqliteError(
            new Database.SqliteError('fixed nonrecoverable condition', code),
          ),
        ).toBe(false);
    } finally {
      database.close();
    }
  });

  it('recognizes a real failed child without classifying programming errors as command failures', () => {
    const error = caught(() =>
      execFileSync(process.execPath, ['-e', 'process.exit(7)'], {
        stdio: 'pipe',
      }),
    );
    expect(isCommandError(error)).toBe(true);
    expect(isCommandError(new TypeError('fixed programming defect'))).toBe(
      false,
    );
    expect(isCommandError(new Error('docker failed'))).toBe(false);
  });

  it('distinguishes a missing file from an unknown exception or malformed errno', () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-errno-'));
    try {
      const error = caught(() =>
        fs.readFileSync(path.join(directory, 'absent')),
      );
      expect(isFileSystemError(error)).toBe(true);
      expect(hasErrno(error, 'ENOENT')).toBe(true);
      expect(hasErrno(error, 'ESRCH')).toBe(false);
      expect(isFileSystemError(new TypeError('fixed defect'))).toBe(false);
      expect(isFileSystemError({ code: 'ENOENT' })).toBe(false);
      expect(
        isFileSystemError(
          Object.assign(new Error('fixed'), { code: 'UNKNOWN' }),
        ),
      ).toBe(false);
    } finally {
      fs.rmSync(directory, { recursive: true });
    }
  });

  it('handles genuine SDK/network failures without swallowing arbitrary errors', () => {
    expect(
      isTelegramError(
        new HttpError('fixed transport failure', new Error('synthetic')),
      ),
    ).toBe(true);
    expect(isTelegramError(new TypeError('fixed handler defect'))).toBe(false);
    expect(
      isNetworkError(
        Object.assign(new Error('fixed reset'), { code: 'ECONNRESET' }),
      ),
    ).toBe(true);
    expect(isNetworkError(new Error('network'))).toBe(false);
  });

  it.each([
    'not cron',
    '70 * * * *',
    '* * * * invalid',
    '*/0 * * * *',
    '* * 40 * *',
  ])(
    'keeps parser validation a recoverable input error for %s',
    (expression) => {
      const error = caught(() =>
        CronExpressionParser.parse(expression, {
          currentDate: '2024-01-01T00:00:00Z',
          tz: 'UTC',
        }),
      );
      expect(isCronInputError(error)).toBe(true);
    },
  );

  it('does not turn unknown parser or programming errors into invalid input', () => {
    expect(isCronInputError(new Error('fixed internal failure'))).toBe(false);
    expect(isCronInputError(new TypeError('Invalid cron expression'))).toBe(
      false,
    );
  });
});
