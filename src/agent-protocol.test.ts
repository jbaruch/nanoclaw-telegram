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
function fixtureHook(
  sdkFailure = false,
  publicationRace: false | 'capture' | 'retire' = false,
  durabilityFault = false,
): string {
  const hook = path.join(directory, 'fixture-hook.mjs');
  // Synthetic filesystem transport only. Linux uses actual proc-FD paths;
  // Darwin maps those fixed Linux capabilities to our owned test directory.
  const fsModule = `import fs from 'node:fs';import path from 'node:path';
    const root=${JSON.stringify(directory)},descriptors=new Map(),events=[];let captured=false,faulted=false;
    const physical=(value)=>{if(typeof value!=='string')return value;
      if(value.startsWith('/workspace/'))return path.join(root,value.slice(1));
      if(value==='/tmp/input.json')return path.join(root,'input.json');
      const match=/^\\/proc\\/self\\/fd\\/([0-9]+)(?:\\/(.*))?$/.exec(value);
      if(match){const held=descriptors.get(Number(match[1]));if(!held)throw new Error('Unknown synthetic directory capability');return path.join(held.path,match[2]??'');}
      return value;};
    const native=(value)=>typeof value==='string'&&value.startsWith('/proc/self/fd/')&&process.platform==='linux'?value:physical(value);
    process.on('exit',()=>fs.writeFileSync(path.join(root,'ipc-durability.json'),JSON.stringify(events)));
    export default new Proxy(fs,{get(target,key){const value=target[key];if(typeof value!=='function')return value;
      return (...args)=>{
        if(key==='renameSync'){
          const destination=physical(args[1]);
          if(${JSON.stringify(publicationRace)}&&!captured&&destination.startsWith(path.join(root,'workspace/ipc/messages')+'/')){
            captured=true;const old=path.join(root,'workspace/ipc/messages'),held=path.join(root,'captured-channel');
            fs.renameSync(old,held);fs.mkdirSync(old,{mode:0o755});
            for(const data of descriptors.values())if(data.path===old||data.path.startsWith(old+'/'))data.path=held+data.path.slice(old.length);
            if(${JSON.stringify(publicationRace)}==='retire')fs.rmdirSync(held);
          }
          if(destination.startsWith(path.join(root,'workspace/ipc')+'/')||destination.startsWith(path.join(root,'captured-channel')+'/'))events.push('rename');
        }
        if(key==='fsyncSync'){
          const held=descriptors.get(args[0]);if(held?.kind){events.push(held.kind+'_sync');
            if(${durabilityFault}&&!faulted&&held.kind==='destination'){faulted=true;throw Object.assign(new Error('Fixed native durability refusal'),{code:'EIO'});}}
        }
        const result=Reflect.apply(value,target,args.map(native));
        if(key==='openSync'){
          const name=physical(args[0]),kind=fs.fstatSync(result).isDirectory()?(name===path.join(root,'workspace/ipc')?'source':name.endsWith('/messages')||name.endsWith('/tasks')?'destination':null):name.startsWith(path.join(root,'workspace/ipc')+'/')?'inode':null;
          descriptors.set(result,{path:name,kind});
        }
        if(key==='closeSync')descriptors.delete(args[0]);
        return result;
      };}});`;
  const source = `import {registerHooks} from 'node:module';
    const fsModule=${JSON.stringify(fsModule)};
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
  publicationRace: false | 'capture' | 'retire' = false,
  durabilityFault = false,
) {
  return spawnSync(
    process.execPath,
    [
      '--import',
      'tsx',
      '--import',
      fixtureHook(sdkFailure, publicationRace, durabilityFault),
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

it.each(['capture', 'retire'] as const)(
  'actual MCP publication survives host channel %s without stranding or duplicating a request',
  (race) => {
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
      race,
    );
    expect(child.error).toBeUndefined();
    expect(child.status, child.stderr).toBe(0);
    const reply = child.stdout
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line))
      .find((row) => row.id === 2);
    expect(
      reply.result.isError ?? false,
      JSON.stringify({
        reply,
        events: JSON.parse(
          fs.readFileSync(path.join(directory, 'ipc-durability.json'), 'utf8'),
        ),
      }),
    ).toBe(false);
    expect(reply.result.content[0].text).toContain('queued');
    const destinations = [
      channel,
      path.join(directory, 'captured-channel'),
    ].filter((dir) => fs.existsSync(dir));
    const published = destinations.flatMap((dir) =>
      fs.readdirSync(dir).map((name) => ({ dir, name })),
    );
    expect(published).toHaveLength(1);
    expect(published[0].name).toMatch(/\.json$/);
    expect(
      JSON.parse(
        fs.readFileSync(path.join(published[0].dir, published[0].name), 'utf8'),
      ).text,
    ).toBe('fixed once');
    const events = JSON.parse(
      fs.readFileSync(path.join(directory, 'ipc-durability.json'), 'utf8'),
    );
    expect(events[0]).toBe('inode_sync');
    expect(events.slice(-3)).toEqual([
      'rename',
      'destination_sync',
      'source_sync',
    ]);
    expect(
      fs
        .readdirSync(path.dirname(channel))
        .filter((name) => name.endsWith('.tmp')),
    ).toEqual([]);
  },
);

it('does not acknowledge queued or replay publication after a native destination fsync refusal', () => {
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
    false,
    true,
  );
  expect(child.error).toBeUndefined();
  expect(child.status, child.stderr).toBe(0);
  const reply = child.stdout
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line))
    .find((row) => row.id === 2);
  expect(reply.result.isError).toBe(true);
  const channel = path.join(directory, 'workspace/ipc/messages'),
    names = fs.readdirSync(channel);
  expect(names).toHaveLength(1);
  expect(
    JSON.parse(fs.readFileSync(path.join(channel, names[0]), 'utf8')).text,
  ).toBe('fixed once');
  expect(
    JSON.parse(
      fs.readFileSync(path.join(directory, 'ipc-durability.json'), 'utf8'),
    ),
  ).toEqual(['inode_sync', 'rename', 'destination_sync']);
});
