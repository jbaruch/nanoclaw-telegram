import { describe, it, expect, vi } from 'vitest';
import { EventEmitter } from 'node:events';
import {
  makeStdioErrorHandler,
  installStdioResilience,
  makeUncaughtEpipeHandler,
  installUncaughtEpipeGuard,
} from './stdio-resilience.js';

const epipe = (): NodeJS.ErrnoException =>
  Object.assign(new Error('write EPIPE'), { code: 'EPIPE' });
const enospc = (): NodeJS.ErrnoException =>
  Object.assign(new Error('no space left on device'), { code: 'ENOSPC' });

describe('makeStdioErrorHandler', () => {
  it('exits 1 on a fatal-channel (stdout) EPIPE', () => {
    const exit = vi.fn() as unknown as ExitFnMock;
    makeStdioErrorHandler(true, exit)(epipe());
    expect(exit).toHaveBeenCalledWith(1);
  });

  it('swallows a non-fatal-channel (stderr) EPIPE without exiting', () => {
    const exit = vi.fn() as unknown as ExitFnMock;
    makeStdioErrorHandler(false, exit)(epipe());
    expect(exit).not.toHaveBeenCalled();
  });

  it('rethrows a non-EPIPE error on the fatal channel without exiting', () => {
    const exit = vi.fn() as unknown as ExitFnMock;
    const handler = makeStdioErrorHandler(true, exit);
    expect(() => handler(enospc())).toThrow('no space left on device');
    expect(exit).not.toHaveBeenCalled();
  });

  it('rethrows a non-EPIPE error on the non-fatal channel', () => {
    const exit = vi.fn() as unknown as ExitFnMock;
    const handler = makeStdioErrorHandler(false, exit);
    expect(() => handler(enospc())).toThrow('no space left on device');
  });
});

describe('installStdioResilience', () => {
  it('exits 1 on stdout EPIPE but leaves the run alive on stderr EPIPE', () => {
    const stdout = new EventEmitter();
    const stderr = new EventEmitter();
    const exit = vi.fn() as unknown as ExitFnMock;
    installStdioResilience(stdout, stderr, exit);

    stderr.emit('error', epipe());
    expect(exit).not.toHaveBeenCalled();

    stdout.emit('error', epipe());
    expect(exit).toHaveBeenCalledTimes(1);
    expect(exit).toHaveBeenCalledWith(1);
  });

  it('swallows an EPIPE event so it never propagates as an unhandled error', () => {
    const stdout = new EventEmitter();
    const stderr = new EventEmitter();
    installStdioResilience(stdout, stderr, vi.fn() as unknown as ExitFnMock);
    expect(() => stderr.emit('error', epipe())).not.toThrow();
  });
});

describe('makeUncaughtEpipeHandler', () => {
  it('exits 0 on a teardown EPIPE after a terminal result was delivered', () => {
    const exit = vi.fn() as unknown as ExitFnMock;
    makeUncaughtEpipeHandler(() => true, exit)(epipe());
    expect(exit).toHaveBeenCalledWith(0);
  });

  it('exits 1 on an EPIPE before any terminal result was delivered', () => {
    const exit = vi.fn() as unknown as ExitFnMock;
    makeUncaughtEpipeHandler(() => false, exit)(epipe());
    expect(exit).toHaveBeenCalledWith(1);
  });

  it('rethrows a non-EPIPE uncaught exception and does not exit', () => {
    const exit = vi.fn() as unknown as ExitFnMock;
    const handler = makeUncaughtEpipeHandler(() => true, exit);
    expect(() => handler(enospc())).toThrow('no space left on device');
    expect(exit).not.toHaveBeenCalled();
  });
});

describe('installUncaughtEpipeGuard', () => {
  it('routes an uncaughtException EPIPE through the handler, exiting per delivery state', () => {
    const proc = new EventEmitter();
    const exit = vi.fn() as unknown as ExitFnMock;
    let delivered = false;
    installUncaughtEpipeGuard(() => delivered, proc, exit);

    proc.emit('uncaughtException', epipe());
    expect(exit).toHaveBeenLastCalledWith(1);

    delivered = true;
    proc.emit('uncaughtException', epipe());
    expect(exit).toHaveBeenLastCalledWith(0);
  });
});

type ExitFnMock = (code: number) => never;
