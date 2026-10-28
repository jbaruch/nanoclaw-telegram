import fs from 'fs';
import os from 'os';
import path from 'path';
import { EventEmitter } from 'events';
import { PassThrough } from 'stream';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
import type { Channel } from './types.js';
import type { DraftStream } from './draft-stream.js';

const fixture = vi.hoisted(() => ({
  directory: '',
  proc: undefined as unknown as EventEmitter & {
    stdin: PassThrough;
    stdout: PassThrough;
    stderr: PassThrough;
    pid: number;
    kill: ReturnType<typeof vi.fn>;
  },
}));
vi.mock('./config.js', () => ({
  ASSISTANT_NAME: 'Andy',
  DEFAULT_TRIGGER: '@Andy',
  getTriggerPattern: () => /^@Andy\b/i,
  get GROUPS_DIR() {
    return path.join(fixture.directory, 'groups');
  },
  get DATA_DIR() {
    return path.join(fixture.directory, 'data');
  },
  get STORE_DIR() {
    return path.join(fixture.directory, 'data');
  },
  CONTAINER_IMAGE: 'synthetic-agent',
  CONTAINER_MAX_OUTPUT_SIZE: 4096,
  CONTAINER_TIMEOUT: 1000,
  IDLE_TIMEOUT: 1000,
  MAX_CONCURRENT_CONTAINERS: 2,
  MAX_MESSAGES_PER_PROMPT: 20,
  ONECLI_URL: 'http://fixture.invalid',
  TIMEZONE: 'UTC',
  POLL_INTERVAL: 1000,
  IPC_POLL_INTERVAL: 1000,
  SCHEDULER_POLL_INTERVAL: 1000,
}));
vi.mock('./logger.js', () => ({
  logger: { debug: vi.fn(), info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));
vi.mock('./channels/index.js', () => ({}));
vi.mock('./mount-security.js', () => ({
  validateAdditionalMounts: vi.fn(() => []),
}));
vi.mock('./container-runtime.js', () => ({
  CONTAINER_RUNTIME_BIN: 'synthetic-docker',
  hostGatewayArgs: () => [],
  readonlyMountArgs: (host: string, container: string) => [
    '-v',
    `${host}:${container}:ro`,
  ],
  stopContainer: vi.fn(),
}));
vi.mock('@onecli-sh/sdk', () => ({
  OneCLI: class {
    applyContainerConfig = vi.fn(async () => true);
    createAgent = vi.fn(async () => ({ id: 'synthetic' }));
    ensureAgent = vi.fn(async () => ({
      name: 'synthetic',
      identifier: 'synthetic',
      created: true,
    }));
  },
}));
vi.mock('child_process', async (original) => ({
  ...(await original<typeof import('child_process')>()),
  spawn: vi.fn(() => fixture.proc),
}));
vi.mock('./router.js', async (original) => ({
  ...(await original<typeof import('./router.js')>()),
  findChannel: vi.fn(),
}));
vi.mock('./db.js', () => ({
  getMessagesSince: vi.fn(),
  getLastBotMessageTimestamp: () => '',
  setRouterState: vi.fn(),
  getAllTasks: () => [],
  getAllChats: () => [],
  setSession: vi.fn(),
}));
import {
  processGroupMessages,
  sendIpcMessage,
  _setRegisteredGroups,
} from './index.js';
import { getMessagesSince, setRouterState } from './db.js';
import { findChannel } from './router.js';
import { stopContainer } from './container-runtime.js';

// Execute the same projection used by the real SDK result handler. The SDK
// event and native channel transport are synthetic; this is not delivery proof.
function sdkOutput(subtype: string, isError: boolean) {
  return JSON.parse(
    execFileSync(
      process.execPath,
      [
        '--import',
        'tsx',
        '--input-type=module',
        '-e',
        `import { sdkResultOutput } from './container/agent-runner/src/sdk-result.ts';
         console.log(JSON.stringify(sdkResultOutput(JSON.parse(process.argv[1]), 'fixed-session')));`,
        JSON.stringify({
          type: 'result',
          subtype,
          is_error: isError,
          result: '',
        }),
      ],
      {
        cwd: fileURLToPath(new URL('../', import.meta.url)),
        encoding: 'utf8',
        timeout: 10000,
        maxBuffer: 65536,
      },
    ),
  ) as Record<string, unknown>;
}

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2024-01-01T00:00:00Z'));
  fixture.directory = fs.mkdtempSync(
    path.join(os.tmpdir(), 'telegram-preview-timeout-'),
  );
  fixture.proc = Object.assign(new EventEmitter(), {
    stdin: new PassThrough(),
    stdout: new PassThrough(),
    stderr: new PassThrough(),
    pid: 12345,
    kill: vi.fn(),
  });
});
afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  fs.rmSync(fixture.directory, { recursive: true, force: true });
});

it.each([
  'preview-timeout',
  'session-timeout',
  'ipc-sdk-complete',
  'sdk-error',
  'sdk-error-close',
  'sdk-is-error',
  'markerless-clean',
  'unfinished-clean',
  'invalid-utf8-clean',
  'session-clean',
  'markerless-overflow',
  'unfinished-overflow',
] as const)(
  'classifies the actual message cursor after real runner %s',
  async (failure) => {
    const jid = `tg:${failure}`;
    _setRegisteredGroups({
      [jid]: {
        name: 'synthetic',
        folder: 'synthetic',
        isMain: true,
        trigger: '@Andy',
        added_at: '2024-01-01T00:00:00Z',
      },
    });
    vi.mocked(getMessagesSince).mockReturnValue([
      {
        id: 'fixed',
        chat_jid: jid,
        sender: 'fixed-owner',
        sender_name: 'fixed',
        content: 'fixed input',
        timestamp: '2024-01-01T00:00:01Z',
      },
    ]);
    const draft: DraftStream = {
      failure: new Promise(() => {}),
      update: vi.fn(async () => {}),
      finish: vi.fn(async () => true),
      cancel: vi.fn(async () => {}),
    };
    const channel: Channel = {
      name: 'synthetic',
      connect: vi.fn(async () => {}),
      disconnect: vi.fn(async () => {}),
      sendMessage: vi.fn(async () => {}),
      isConnected: () => true,
      ownsJid: () => true,
      setTyping: vi.fn(async () => {}),
      createDraftStream: () => draft,
    };
    vi.mocked(findChannel).mockReturnValue(channel);
    const pending = processGroupMessages(jid);
    await vi.advanceTimersByTimeAsync(10);
    if (
      failure === 'preview-timeout' ||
      failure === 'session-timeout' ||
      failure === 'ipc-sdk-complete' ||
      failure === 'sdk-error' ||
      failure === 'sdk-error-close' ||
      failure === 'sdk-is-error'
    ) {
      if (failure === 'ipc-sdk-complete')
        await sendIpcMessage(jid, 'fixed queued reply');
      const output =
        failure === 'preview-timeout'
          ? { status: 'success', result: null, streamText: 'fixed preview' }
          : failure === 'session-timeout'
            ? {
                status: 'success',
                result: null,
                newSessionId: 'fixed-session',
              }
            : sdkOutput(
                failure.startsWith('sdk-error')
                  ? 'error_during_execution'
                  : 'success',
                failure !== 'ipc-sdk-complete',
              );
      if (failure === 'ipc-sdk-complete') {
        expect(output).toMatchObject({
          status: 'success',
          result: null,
          completion: 'sdk_turn_completed',
        });
      } else if (
        failure.startsWith('sdk-error') ||
        failure === 'sdk-is-error'
      ) {
        expect(output.status).toBe('error');
        expect(output).not.toHaveProperty('completion');
      }
      fixture.proc.stdout.push(
        '---NANOCLAW_OUTPUT_START---\n' +
          JSON.stringify(output) +
          '\n---NANOCLAW_OUTPUT_END---\n',
      );
      await vi.advanceTimersByTimeAsync(1);
      if (failure === 'sdk-error-close') fixture.proc.emit('close', 1);
      else {
        await vi.advanceTimersByTimeAsync(31001);
        fixture.proc.emit('close', 137);
      }
    } else if (failure.endsWith('-clean')) {
      const session = JSON.stringify({
        status: 'success',
        result: null,
        newSessionId: 'fixed-session',
      });
      fixture.proc.stdout.push(
        failure === 'invalid-utf8-clean'
          ? Buffer.concat([
              Buffer.from(
                '---NANOCLAW_OUTPUT_START---\n{"status":"success","result":"',
              ),
              Buffer.from([0xff]),
              Buffer.from('"}\n---NANOCLAW_OUTPUT_END---\n'),
            ])
          : failure === 'markerless-clean'
            ? 'fixed diagnostic only'
            : '---NANOCLAW_OUTPUT_START---\n' +
              session +
              (failure === 'session-clean'
                ? '\n---NANOCLAW_OUTPUT_END---\n'
                : ''),
      );
      await vi.advanceTimersByTimeAsync(0);
      fixture.proc.emit('close', 0);
    } else {
      fixture.proc.stdout.push(
        (failure === 'unfinished-overflow'
          ? '---NANOCLAW_OUTPUT_START---'
          : '') + 'x'.repeat(4097),
      );
      await vi.advanceTimersByTimeAsync(0);
    }
    await expect(pending).resolves.toBe(failure === 'ipc-sdk-complete');
    if (failure === 'preview-timeout')
      expect(draft.update).toHaveBeenCalledExactlyOnceWith('fixed preview');
    else expect(draft.update).not.toHaveBeenCalled();
    expect(draft.finish).not.toHaveBeenCalled();
    expect(draft.cancel).toHaveBeenCalledTimes(1);
    const writes = vi
      .mocked(setRouterState)
      .mock.calls.filter(([key]) => key === 'last_agent_timestamp');
    expect(JSON.parse(writes.at(-1)![1])[jid]).toBe(
      failure === 'ipc-sdk-complete' ? '2024-01-01T00:00:01Z' : '',
    );
    if (failure === 'ipc-sdk-complete')
      expect(channel.sendMessage).toHaveBeenCalledExactlyOnceWith(
        jid,
        'fixed queued reply',
        undefined,
      );
    else expect(channel.sendMessage).not.toHaveBeenCalled();
    expect(channel.setTyping).toHaveBeenLastCalledWith(jid, false);
    if (failure.endsWith('-clean') || failure === 'sdk-error-close')
      expect(stopContainer).not.toHaveBeenCalled();
    else
      expect(stopContainer).toHaveBeenCalledExactlyOnceWith(
        expect.stringContaining('nanoclaw-synthetic-'),
      );
    expect(vi.getTimerCount()).toBe(0);
  },
);
