import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawnSync } from 'child_process';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
const context = vi.hoisted(() => ({ timezone: 'UTC' }));
vi.mock('./config.js', () => ({
  get TIMEZONE() {
    return context.timezone;
  },
}));
import { parseTaskSchedule } from './task-schedule.js';
let directory: string;
beforeEach(() => {
  context.timezone = 'UTC';
  vi.stubEnv('TZ', 'UTC');
  directory = fs.mkdtempSync(
    path.join(os.tmpdir(), 'telegram-agent-protocol-'),
  );
});
afterEach(() => {
  vi.unstubAllEnvs();
  fs.rmSync(directory, { recursive: true, force: true });
});
function fixtureHook(sdkFailure = false, publicationRace = false): string {
  const hook = path.join(directory, 'fixture-hook.mjs');
  // Test-only native transport substitutions. Production code has no injected
  // path/SDK selectors; the unmodified actual entrypoints execute below.
  const source = `import {registerHooks} from 'node:module';
    const fsModule=${JSON.stringify(`import fs from 'node:fs';import path from 'node:path';
      const root=${JSON.stringify(directory)};let captured=false;
      export default new Proxy(fs,{get(target,key){const value=target[key];if(typeof value!=='function')return value;
        return (...args)=>{if(${publicationRace}&&!captured&&key==='renameSync'&&typeof args[1]==='string'&&args[1].startsWith('/workspace/ipc/messages/')){captured=true;fs.renameSync(path.join(root,'workspace/ipc/messages'),path.join(root,'captured-channel'));}
        return Reflect.apply(value,target,args.map(arg=>typeof arg==='string'&&arg.startsWith('/workspace/')?path.join(root,arg.slice(1)):arg==='/tmp/input.json'?path.join(root,'input.json'):arg));};}});`)};
    const sdk=${JSON.stringify("setInterval(()=>{},60000); export async function* query(){yield {type:'result',subtype:'error_during_execution',is_error:true,errors:['fixed-private-sdk-body']};throw new Error('SDK event after failure must not be consumed');}")};
    registerHooks({resolve(specifier,context,next){
      if(specifier==='fs')return {url:'data:text/javascript,'+encodeURIComponent(fsModule),shortCircuit:true};
      if(${sdkFailure}&&specifier==='@anthropic-ai/claude-agent-sdk')return {url:'data:text/javascript,'+encodeURIComponent(sdk),shortCircuit:true};
      return next(specifier,context);
    }});`;
  fs.writeFileSync(hook, source);
  return hook;
}
function run(
  file: string,
  input: string,
  sdkFailure = false,
  timezone = 'UTC',
  publicationRace = false,
) {
  return spawnSync(
    process.execPath,
    [
      '--import',
      'tsx',
      '--import',
      fixtureHook(sdkFailure, publicationRace),
      fileURLToPath(
        new URL('../container/agent-runner/src/' + file, import.meta.url),
      ),
    ],
    {
      input,
      encoding: 'utf8',
      timeout: 5000,
      maxBuffer: 262144,
      cwd: fileURLToPath(new URL('../', import.meta.url)),
      env: {
        PATH: process.env.PATH ?? '',
        HOME: directory,
        TZ: timezone,
        NANOCLAW_CHAT_JID: 'tg:fixture',
        NANOCLAW_GROUP_FOLDER: 'main',
        NANOCLAW_IS_MAIN: '1',
      },
    },
  );
}
it('the actual main SDK failure exits immediately without a success/session completion or leaked body', () => {
  const child = run(
    'index.ts',
    JSON.stringify({
      prompt: 'fixed',
      chatJid: 'tg:fixture',
      groupFolder: 'main',
      isMain: true,
    }),
    true,
  );
  expect(child.error).toBeUndefined();
  expect(child.status).toBe(1);
  const frames = [
    ...child.stdout.matchAll(
      /---NANOCLAW_OUTPUT_START---\s*([\s\S]*?)\s*---NANOCLAW_OUTPUT_END---/g,
    ),
  ].map((match) => JSON.parse(match[1]));
  expect(frames).toEqual([
    {
      status: 'error',
      result: null,
      error: 'SDK query did not complete successfully (error_during_execution)',
    },
  ]);
  expect(child.stderr).not.toContain('fixed-private-sdk-body');
  expect(child.stderr).not.toContain('SDK event after failure');
});
it.each([
  ['once', '2026-02-01T15:30', false, 'UTC'],
  ['once', '2026-02-01', false, 'UTC'],
  ['once', '2026-02-01 15:30:00', false, 'UTC'],
  ['once', '2026-02-30T15:30:00', false, 'UTC'],
  ['interval', '5000ms', false, 'UTC'],
  ['interval', '5.5', false, 'UTC'],
  ['interval', '1e3', false, 'UTC'],
  ['interval', '5000', true, 'UTC'],
  ['once', '2026-02-01T15:30:00', true, 'UTC'],
  ['once', '2026-02-01T15:30:00Z', true, 'UTC'],
  ['once', '2026-03-08T02:30:00', false, 'America/Chicago'],
  ['once', '2025-11-02T01:30:00', false, 'America/Chicago'],
  ['once', '2025-11-02T01:30:00-05:00', true, 'America/Chicago'],
] as const)(
  'actual MCP create/update agrees with host grammar for %s %s',
  (type, value, valid, timezone) => {
    context.timezone = timezone;
    fs.mkdirSync(path.join(directory, 'workspace', 'ipc'), { recursive: true });
    fs.writeFileSync(
      path.join(directory, 'workspace', 'ipc', 'current_tasks.json'),
      JSON.stringify([
        { id: 'fixed-task', schedule_type: type, schedule_value: value },
      ]),
    );
    const requests = [
      {
        jsonrpc: '2.0',
        id: 1,
        method: 'initialize',
        params: {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: { name: 'synthetic', version: '1' },
        },
      },
      { jsonrpc: '2.0', method: 'notifications/initialized' },
      {
        jsonrpc: '2.0',
        id: 2,
        method: 'tools/call',
        params: {
          name: 'schedule_task',
          arguments: {
            prompt: 'fixed',
            schedule_type: type,
            schedule_value: value,
          },
        },
      },
      {
        jsonrpc: '2.0',
        id: 3,
        method: 'tools/call',
        params: {
          name: 'update_task',
          arguments: {
            task_id: 'fixed-task',
            schedule_type: type,
            schedule_value: value,
          },
        },
      },
      {
        jsonrpc: '2.0',
        id: 4,
        method: 'tools/call',
        params: {
          name: 'update_task',
          arguments: { task_id: 'fixed-task', schedule_value: value },
        },
      },
    ];
    const child = run(
      'ipc-mcp-stdio.ts',
      requests.map((row) => JSON.stringify(row)).join('\n') + '\n',
      false,
      timezone,
    );
    expect(child.error).toBeUndefined();
    expect(child.status, child.stderr).toBe(0);
    const replies = child.stdout
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));
    for (const id of [2, 3, 4]) {
      const reply = replies.find((row) => row.id === id);
      expect(reply).toBeDefined();
      expect(reply.result.isError ?? false).toBe(!valid);
    }
    if (valid) expect(() => parseTaskSchedule(type, value)).not.toThrow();
    else expect(() => parseTaskSchedule(type, value)).toThrow(SyntaxError);
    const tasks = path.join(directory, 'workspace', 'ipc', 'tasks');
    expect(fs.existsSync(tasks) ? fs.readdirSync(tasks).length : 0).toBe(
      valid ? 3 : 0,
    );
  },
);

it('interprets local once time in configured Chicago while the native host process is UTC', () => {
  context.timezone = 'America/Chicago';
  expect(new Date('2026-02-01T15:30:00').toISOString()).toBe(
    '2026-02-01T15:30:00.000Z',
  );
  expect(parseTaskSchedule('once', '2026-02-01T15:30:00').nextRun).toBe(
    '2026-02-01T21:30:00.000Z',
  );
  expect(() => parseTaskSchedule('once', '2026-03-08T02:30:00')).toThrow(
    'nonexistent or ambiguous',
  );
  expect(() => parseTaskSchedule('once', '2025-11-02T01:30:00')).toThrow(
    'nonexistent or ambiguous',
  );
  expect(parseTaskSchedule('once', '2025-11-02T01:30:00-05:00').nextRun).toBe(
    '2025-11-02T06:30:00.000Z',
  );
});

it('actual MCP publication survives a host channel capture without stranding a temporary file or duplicating the request', () => {
  const channel = path.join(directory, 'workspace', 'ipc', 'messages');
  fs.mkdirSync(channel, { recursive: true });
  const requests = [
    {
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'synthetic', version: '1' },
      },
    },
    { jsonrpc: '2.0', method: 'notifications/initialized' },
    {
      jsonrpc: '2.0',
      id: 2,
      method: 'tools/call',
      params: { name: 'send_message', arguments: { text: 'fixed once' } },
    },
  ];
  const child = run(
    'ipc-mcp-stdio.ts',
    requests.map((row) => JSON.stringify(row)).join('\n') + '\n',
    false,
    'UTC',
    true,
  );
  expect(child.error).toBeUndefined();
  expect(child.status, child.stderr).toBe(0);
  const reply = child.stdout
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line))
    .find((row) => row.id === 2);
  expect(reply.result.isError ?? false).toBe(false);
  expect(reply.result.content[0].text).toContain('queued');
  expect(fs.readdirSync(path.join(directory, 'captured-channel'))).toEqual([]);
  const names = fs.readdirSync(channel);
  expect(names).toHaveLength(1);
  expect(names[0]).toMatch(/\.json$/);
  expect(
    JSON.parse(fs.readFileSync(path.join(channel, names[0]), 'utf8')).text,
  ).toBe('fixed once');
  expect(
    fs
      .readdirSync(path.dirname(channel))
      .filter((name) => name.endsWith('.tmp')),
  ).toEqual([]);
});
