import { describe, it, expect, afterEach, vi } from 'vitest';

import {
  _resetLocationSinksForTests,
  registerLocationSink,
  runLocationSinks,
} from './location-sinks.js';
import { logger } from './logger.js';
import type { LocationRecord } from './types.js';

const RECORD: LocationRecord = {
  chat_jid: 'tg:123',
  sender: 'owner-1',
  message_id: 'msg-1',
  latitude: 41.03,
  longitude: -73.76,
  source: 'static',
  recorded_at: '2026-07-21T12:00:00.000Z',
};

afterEach(() => {
  // The sink list is module-global shared state — wipe it so no test's
  // registration leaks into another and order never matters.
  _resetLocationSinksForTests();
  vi.restoreAllMocks();
});

describe('location sink registry', () => {
  it('fans a record out to every sink in registration order', () => {
    const calls: string[] = [];
    registerLocationSink('a', (r) => {
      calls.push(`a:${r.sender}`);
    });
    registerLocationSink('b', (r) => {
      calls.push(`b:${r.sender}`);
    });
    runLocationSinks(RECORD);
    expect(calls).toEqual(['a:owner-1', 'b:owner-1']);
  });

  it('is a no-op with nothing registered (platform-only install)', () => {
    expect(() => runLocationSinks(RECORD)).not.toThrow();
  });

  it('throws on duplicate sink names', () => {
    registerLocationSink('dup', () => {});
    expect(() => registerLocationSink('dup', () => {})).toThrow(
      /already registered: dup/,
    );
  });

  it('routes an async sink rejection to the error log (no unhandled rejection)', async () => {
    const errorLog = vi.spyOn(logger, 'error').mockImplementation(() => {});
    const after = vi.fn();
    registerLocationSink('async-boom', async () => {
      throw new Error('async artifact write failed');
    });
    registerLocationSink('after', after);
    runLocationSinks(RECORD);
    expect(after).toHaveBeenCalledOnce();
    // Let the rejected promise's .catch handler run.
    await new Promise((resolve) => setImmediate(resolve));
    expect(errorLog).toHaveBeenCalledWith(
      expect.objectContaining({ sink: 'async-boom' }),
      'Location sink failed',
    );
  });

  it('isolates a throwing sink: logs it and still runs the rest', () => {
    const errorLog = vi.spyOn(logger, 'error').mockImplementation(() => {});
    const after = vi.fn();
    registerLocationSink('boom', () => {
      throw new Error('artifact write failed');
    });
    registerLocationSink('after', after);
    runLocationSinks(RECORD);
    expect(after).toHaveBeenCalledOnce();
    expect(errorLog).toHaveBeenCalledWith(
      expect.objectContaining({ sink: 'boom' }),
      'Location sink failed',
    );
  });
});
