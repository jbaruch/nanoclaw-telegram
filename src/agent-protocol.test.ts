import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import { ChildProcess, spawnSync } from 'child_process';
import { GroupQueue } from './group-queue.js';
import { afterEach, beforeEach, expect, it, vi } from 'vitest';
const context = vi.hoisted(() => ({ timezone: 'UTC', data: '' }));
vi.mock('@onecli-sh/sdk', () => ({ OneCLI: class {} }));
vi.mock('./config.js', () => ({
  ONECLI_URL: 'http://fixture.invalid',
  MAX_CONCURRENT_CONTAINERS: 2,
  get DATA_DIR() {
    return context.data;
  },
  get GROUPS_DIR() {
    return path.join(context.data, 'groups');
  },
  get TIMEZONE() {
    return context.timezone;
  },
}));
import { writeTasksSnapshot } from './container-runner.js';
import { parseTaskSchedule } from './task-schedule.js';
let directory: string;
beforeEach(() => {
  context.timezone = 'UTC';
  vi.stubEnv('TZ', 'UTC');
  directory = fs.mkdtempSync(
    path.join(os.tmpdir(), 'telegram-agent-protocol-'),
  );
  context.data = path.join(directory, 'host-data');
});
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  fs.rmSync(directory, { recursive: true, force: true });
});
function fixtureHook(
  sdkFailure:
    | boolean
    | 'throw'
    | 'drain'
    | 'poll-failure'
    | 'wait-failure'
    | 'settle-poll-failure' = false,
  publicationRace: false | 'capture' | 'retire' = false,
  durabilityFault = false,
): string {
  const hook = path.join(directory, 'fixture-hook.mjs');
  // Synthetic filesystem transport only. Linux uses actual proc-FD paths;
  // Darwin maps those fixed Linux capabilities to our owned test directory.
  const fsModule = `import fs from 'node:fs';import path from 'node:path';
    const root=${JSON.stringify(directory)},descriptors=new Map(),events=[];let captured=false,faulted=false,inputFaulted=false,closeFaulted=false;
    const physical=(value)=>{if(typeof value!=='string')return value;
      if(value.startsWith('/workspace/'))return path.join(root,value.slice(1));
      if(value==='/tmp/input.json')return path.join(root,'input.json');
      if(value==='/tmp/task-script.sh')return path.join(root,'task-script.sh');
      const match=/^\\/proc\\/self\\/fd\\/([0-9]+)(?:\\/(.*))?$/.exec(value);
      if(match){const held=descriptors.get(Number(match[1]));if(!held)throw new Error('Unknown synthetic directory capability');return path.join(held.path,match[2]??'');}
      return value;};
    const native=(value)=>typeof value==='string'&&value.startsWith('/proc/self/fd/')&&process.platform==='linux'?value:physical(value);
    process.on('exit',()=>fs.writeFileSync(path.join(root,'ipc-durability.json'),JSON.stringify(events)));
    export default new Proxy(fs,{get(target,key){const value=target[key];if(typeof value!=='function')return value;
      return (...args)=>{
        if(${JSON.stringify(sdkFailure === 'drain')}&&key==='readFileSync'&&String(args[0]).endsWith('/input/002-held.json')&&!inputFaulted){inputFaulted=true;throw Object.assign(new Error('Fixed later native input read failure'),{code:'EIO'});}
        if(${JSON.stringify(sdkFailure === 'poll-failure' || sdkFailure === 'wait-failure' || sdkFailure === 'settle-poll-failure')}&&key==='unlinkSync'&&String(args[0]).endsWith('/input/_close')&&fs.existsSync(physical(args[0]))&&!closeFaulted){closeFaulted=true;globalThis.fixedFixtureSDKDone?.();throw Object.assign(new Error('Fixed private native close failure'),{code:'EACCES'});}
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
  const childModule = `import * as native from 'node:child_process';export * from 'node:child_process';export function execFile(file,args,options,callback){return native.execFile(file,args.map(value=>value==='/tmp/task-script.sh'?${JSON.stringify(path.join(directory, 'task-script.sh'))}:value),options,callback);}`;
  const source = `import {registerHooks} from 'node:module';
    if(${JSON.stringify(sdkFailure === 'throw')}){const write=process.stdout.write.bind(process.stdout);process.stdout.write=(...args)=>{setTimeout(()=>write(...args),50);return true;};}
    const fsModule=${JSON.stringify(fsModule)};
    const childModule=${JSON.stringify(childModule)};
    const sdk=${JSON.stringify(sdkFailure === 'settle-poll-failure' ? "import fs from 'fs';export function query(){let step=0;return {[Symbol.asyncIterator](){return this;},next(){if(++step===1)return Promise.resolve({done:false,value:{type:'result',subtype:'success',is_error:false,result:'initial complete'}});fs.writeFileSync('/workspace/ipc/input/_close','close');return new Promise(resolve=>{globalThis.fixedFixtureSDKDone=()=>resolve({done:true});});}};}" : sdkFailure === 'poll-failure' ? "import fs from 'fs';export async function* query(){fs.writeFileSync('/workspace/ipc/input/_close','close');await new Promise(()=>{});}" : sdkFailure === 'wait-failure' ? "import fs from 'fs';export async function* query(){fs.writeFileSync('/workspace/ipc/input/_close','close');yield {type:'result',subtype:'success',is_error:false,result:'initial complete'};}" : sdkFailure === 'drain' ? "import fs from 'fs';export async function* query({prompt}){const first=await prompt[Symbol.asyncIterator]().next();fs.writeFileSync('/workspace/ipc/input/_close','close');yield {type:'result',subtype:'success',is_error:false,result:first.value.message.content};}" : sdkFailure === 'throw' ? "export async function* query(){throw new TypeError('fixed-private-unexpected-sdk-body');}" : "setInterval(()=>{},60000); export async function* query(){yield {type:'result',subtype:'error_during_execution',is_error:true,errors:['fixed-private-sdk-body']};throw new Error('SDK event after failure must not be consumed');}")};
    registerHooks({resolve(specifier,context,next){
      if(specifier==='fs')return {url:'data:text/javascript,'+encodeURIComponent(fsModule),shortCircuit:true};
      if(specifier==='child_process')return {url:'data:text/javascript,'+encodeURIComponent(childModule),shortCircuit:true};
      if(${Boolean(sdkFailure)}&&specifier==='@anthropic-ai/claude-agent-sdk')return {url:'data:text/javascript,'+encodeURIComponent(sdk),shortCircuit:true};
      return next(specifier,context);
    }});`;
  fs.writeFileSync(hook, source);
  return hook;
}

function run(
  file: string,
  input: string,
  sdkFailure:
    | boolean
    | 'throw'
    | 'drain'
    | 'poll-failure'
    | 'wait-failure'
    | 'settle-poll-failure' = false,
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
      JSON.stringify({
        schema_version: 1,
        tasks: [
          {
            id: 'fixed-task',
            groupFolder: 'main',
            prompt: 'fixed',
            script: null,
            status: 'active',
            next_run: null,
            schedule_type: type,
            schedule_value: value,
          },
        ],
      }),
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
    JSON.parse(fs.readFileSync(path.join(channel, names[0]), 'utf8')),
  ).toMatchObject({ schema_version: 1, type: 'message', text: 'fixed once' });
  expect(
    JSON.parse(
      fs.readFileSync(path.join(directory, 'ipc-durability.json'), 'utf8'),
    ),
  ).toEqual(['inode_sync', 'rename', 'destination_sync']);
});

function snapshotCalls(): string {
  return (
    [
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
        params: { name: 'list_tasks', arguments: {} },
      },
      {
        jsonrpc: '2.0',
        id: 3,
        method: 'tools/call',
        params: {
          name: 'update_task',
          arguments: { task_id: 'fixed-task', schedule_value: '60000' },
        },
      },
    ]
      .map((value) => JSON.stringify(value))
      .join('\n') + '\n'
  );
}
const snapshotRow = {
  id: 'fixed-task',
  groupFolder: 'main',
  prompt: 'fixed prompt',
  script: null,
  schedule_type: 'interval',
  schedule_value: '5000',
  status: 'active',
  next_run: null,
};

it('the actual host snapshot writer feeds the actual MCP list and partial-update reader', () => {
  writeTasksSnapshot('main', true, [snapshotRow]);
  const nativeFile = path.join(
    context.data,
    'ipc',
    'main',
    'current_tasks.json',
  );
  const bytes = fs.readFileSync(nativeFile);
  expect(JSON.parse(bytes.toString())).toEqual({
    schema_version: 1,
    tasks: [snapshotRow],
  });
  const mounted = path.join(
    directory,
    'workspace',
    'ipc',
    'current_tasks.json',
  );
  fs.mkdirSync(path.dirname(mounted), { recursive: true });
  fs.writeFileSync(mounted, bytes);
  const child = run('ipc-mcp-stdio.ts', snapshotCalls());
  expect(child.error).toBeUndefined();
  expect(child.status, child.stderr).toBe(0);
  const replies = child.stdout
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  for (const id of [2, 3])
    expect(
      replies.find((reply) => reply.id === id).result.isError ?? false,
    ).toBe(false);
  const requests = path.join(directory, 'workspace', 'ipc', 'tasks');
  expect(fs.readdirSync(requests)).toHaveLength(1);
  expect(fs.readFileSync(mounted)).toEqual(bytes);
});

it.each([
  [snapshotRow],
  { schema_version: 2, tasks: [snapshotRow] },
  { schema_version: 1, tasks: [snapshotRow], extra: true },
  { schema_version: 1, tasks: [snapshotRow, snapshotRow] },
  { schema_version: 1, tasks: [{ ...snapshotRow, script: 123 }] },
  { schema_version: 1, tasks: [{ id: 'fixed-task' }] },
  { schema_version: 1, tasks: null },
])(
  'actual MCP refuses incompatible snapshot %j without IPC publication or mutation',
  (snapshot) => {
    const file = path.join(directory, 'workspace', 'ipc', 'current_tasks.json');
    fs.mkdirSync(path.dirname(file), { recursive: true });
    const bytes = JSON.stringify(snapshot);
    fs.writeFileSync(file, bytes);
    const child = run('ipc-mcp-stdio.ts', snapshotCalls());
    expect(child.error).toBeUndefined();
    expect(child.status, child.stderr).toBe(0);
    const replies = child.stdout
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));
    for (const id of [2, 3])
      expect(replies.find((reply) => reply.id === id).result.isError).toBe(
        true,
      );
    expect(
      fs.existsSync(path.join(directory, 'workspace', 'ipc', 'tasks')),
    ).toBe(false);
    expect(fs.readFileSync(file, 'utf8')).toBe(bytes);
  },
);

it('routes an actual unexpected SDK producer defect through the sole outer framed process boundary', () => {
  const child = run(
    'index.ts',
    JSON.stringify({
      prompt: 'fixed',
      chatJid: 'tg:fixture',
      groupFolder: 'main',
      isMain: true,
    }),
    'throw',
  );
  expect(child.error).toBeUndefined();
  expect(child.status).toBe(1);
  expect(child.stdout).toContain('---NANOCLAW_OUTPUT_START---');
  expect(child.stdout).toContain('Agent execution failed');
  expect(child.stdout + child.stderr).not.toContain(
    'fixed-private-unexpected-sdk-body',
  );
});

it.each([
  'null',
  '[]',
  '{"wakeAgent":"false"}',
  '{broken',
  '{"wakeAgent":true,"data":null}',
  '{"wakeAgent":true,"data":[]}',
  '{"wakeAgent":true,"data":7}',
  '{"wakeAgent":true,"data":"private"}',
])(
  'actual scheduled precheck refuses malformed output %s without querying the SDK',
  (output) => {
    const script = 'printf "%s\\n" ' + "'" + output + "'";
    const child = run(
      'index.ts',
      JSON.stringify({
        prompt: 'fixed',
        chatJid: 'tg:fixture',
        groupFolder: 'main',
        isMain: true,
        isScheduledTask: true,
        script,
      }),
      'throw',
    );
    expect(child.error).toBeUndefined();
    expect(child.status).toBe(0); // Logical precheck failure is an error frame, not an invented native exit.
    expect(child.stdout).toContain(
      'Task script did not return a valid precheck result',
    );
    expect(child.stdout + child.stderr).not.toContain('fixed-private-sdk-body');
  },
);

it('host snapshot replacement never writes through an agent-controlled leaf symlink', () => {
  const group = path.join(context.data, 'ipc', 'main');
  fs.mkdirSync(group, { recursive: true });
  const outside = path.join(directory, 'retained-owner-file');
  fs.writeFileSync(outside, 'unchanged owner bytes');
  fs.symlinkSync(outside, path.join(group, 'current_tasks.json'));
  writeTasksSnapshot('main', false, [
    snapshotRow,
    { ...snapshotRow, id: 'other-task', groupFolder: 'other' },
  ]);
  expect(fs.readFileSync(outside, 'utf8')).toBe('unchanged owner bytes');
  expect(
    fs.lstatSync(path.join(group, 'current_tasks.json')).isSymbolicLink(),
  ).toBe(false);
  expect(
    JSON.parse(fs.readFileSync(path.join(group, 'current_tasks.json'), 'utf8')),
  ).toEqual({ schema_version: 1, tasks: [snapshotRow] });
  expect(
    fs
      .readdirSync(path.dirname(group))
      .filter((name) => name.startsWith('.tasks-snapshot-')),
  ).toEqual([]);
});

it('host snapshot writer refuses a group alias without changing foreign bytes', () => {
  const base = path.join(context.data, 'ipc');
  fs.mkdirSync(base, { recursive: true });
  const foreign = path.join(directory, 'foreign');
  fs.mkdirSync(foreign);
  fs.writeFileSync(path.join(foreign, 'retained'), 'unchanged');
  fs.symlinkSync(foreign, path.join(base, 'main'));
  expect(() => writeTasksSnapshot('main', true, [snapshotRow])).toThrow(
    'Task snapshot parent is unsafe',
  );
  expect(fs.readdirSync(foreign)).toEqual(['retained']);
  expect(fs.readFileSync(path.join(foreign, 'retained'), 'utf8')).toBe(
    'unchanged',
  );
});

it('failed snapshot replacement preserves the prior artifact and reports cleanup defects', () => {
  writeTasksSnapshot('main', true, [snapshotRow]);
  const file = path.join(context.data, 'ipc', 'main', 'current_tasks.json'),
    before = fs.readFileSync(file);
  const error = Object.assign(new Error('fixed snapshot replace failure'), {
    code: 'EIO',
  });
  const rename = vi.spyOn(fs, 'renameSync').mockImplementationOnce(() => {
    throw error;
  });
  expect(() =>
    writeTasksSnapshot('main', true, [{ ...snapshotRow, prompt: 'new' }]),
  ).toThrow(error);
  expect(fs.readFileSync(file)).toEqual(before);
  rename.mockRestore();
  const cleanup = Object.assign(new Error('fixed snapshot cleanup failure'), {
    code: 'EACCES',
  });
  const remove = vi.spyOn(fs, 'rmSync').mockImplementationOnce(() => {
    throw cleanup;
  });
  expect(() => writeTasksSnapshot('main', true, [snapshotRow])).toThrow(
    cleanup,
  );
  remove.mockRestore();
});

it('a prior open snapshot descriptor cannot mutate its atomically published successor', () => {
  writeTasksSnapshot('main', true, [snapshotRow]);
  const file = path.join(context.data, 'ipc', 'main', 'current_tasks.json');
  const fd = fs.openSync(file, 'r+');
  using _lease = { [Symbol.dispose]: () => fs.closeSync(fd) };
  writeTasksSnapshot('main', true, [
    { ...snapshotRow, prompt: 'fresh revision' },
  ]);
  fs.writeSync(fd, Buffer.from('invalid old bytes'), 0, 17, 0);
  expect(JSON.parse(fs.readFileSync(file, 'utf8')).tasks[0].prompt).toBe(
    'fresh revision',
  );
});

it('snapshot cleanup failure preserves the original failed-replace error too', () => {
  writeTasksSnapshot('main', true, [snapshotRow]);
  const primary = Object.assign(new Error('fixed snapshot replace failure'), {
      code: 'EIO',
    }),
    cleanup = Object.assign(new Error('fixed snapshot cleanup failure'), {
      code: 'EACCES',
    });
  vi.spyOn(fs, 'renameSync').mockImplementationOnce(() => {
    throw primary;
  });
  vi.spyOn(fs, 'rmSync').mockImplementationOnce(() => {
    throw cleanup;
  });
  let failure: Error | undefined;
  try {
    writeTasksSnapshot('main', true, [snapshotRow]);
  } catch (error) {
    if (!(error instanceof Error) || error.name !== 'SuppressedError')
      throw error;
    failure = error;
  }
  expect(failure).toMatchObject({
    name: 'SuppressedError',
    error: cleanup,
    suppressed: primary,
  });
});

it('actual agent drain keeps already consumed input when a later native read fails', () => {
  const input = path.join(directory, 'workspace', 'ipc', 'input');
  fs.mkdirSync(input, { recursive: true });
  fs.writeFileSync(
    path.join(input, '001-first.json'),
    JSON.stringify({
      schema_version: 1,
      type: 'message',
      text: 'first synthetic input',
    }),
  );
  const held = JSON.stringify({
    schema_version: 1,
    type: 'message',
    text: 'later held synthetic input',
  });
  fs.writeFileSync(
    path.join(input, '003-independent.json'),
    JSON.stringify({
      schema_version: 1,
      type: 'message',
      text: 'independent synthetic input',
    }),
  );
  fs.writeFileSync(path.join(input, '002-held.json'), held);
  const child = run(
    'index.ts',
    JSON.stringify({
      prompt: 'initial fixed prompt',
      chatJid: 'tg:fixture',
      groupFolder: 'main',
      isMain: true,
    }),
    'drain',
  );
  expect(child.error).toBeUndefined();
  expect(child.status, child.stderr).toBe(0);
  const frames = [
    ...child.stdout.matchAll(
      /---NANOCLAW_OUTPUT_START---\s*([\s\S]*?)\s*---NANOCLAW_OUTPUT_END---/g,
    ),
  ].map((match) => JSON.parse(match[1]));
  expect(
    frames.find((frame) => frame.completion === 'sdk_turn_completed')?.result,
  ).toBe(
    'initial fixed prompt\nfirst synthetic input\nindependent synthetic input',
  );
  expect(fs.existsSync(path.join(input, '001-first.json'))).toBe(false);
  expect(fs.readFileSync(path.join(input, '002-held.json'), 'utf8')).toBe(held);
  expect(child.stderr).toContain('IPC input read failed');
});

it.each(['poll-failure', 'wait-failure', 'settle-poll-failure'] as const)(
  'actual %s reaches the awaited framed agent boundary',
  (failure) => {
    const child = run(
      'index.ts',
      JSON.stringify({
        prompt: 'fixed',
        chatJid: 'tg:fixture',
        groupFolder: 'main',
        isMain: true,
      }),
      failure,
    );
    expect(child.error).toBeUndefined();
    expect(child.status).toBe(1);
    const frames = [
      ...child.stdout.matchAll(
        /---NANOCLAW_OUTPUT_START---\s*([\s\S]*?)\s*---NANOCLAW_OUTPUT_END---/g,
      ),
    ].map((match) => JSON.parse(match[1]));
    expect(frames.at(-1)).toMatchObject({
      status: 'error',
      result: null,
      error: expect.stringContaining('Agent IPC/storage failed'),
    });
    expect(child.stdout + child.stderr).not.toContain(
      'Fixed private native close failure',
    );
    expect(
      fs.existsSync(
        path.join(directory, 'workspace', 'ipc', 'input', '_close'),
      ),
    ).toBe(true);
  },
);

it('actual host follow-up writer and agent reader share schema1 while incompatible input stays unchanged', async () => {
  const input = path.join(directory, 'workspace', 'ipc', 'input');
  const hostInput = path.join(context.data, 'ipc', 'main', 'input');
  fs.mkdirSync(input, { recursive: true });
  const incompatible = [
    { type: 'message', text: 'legacy private input' },
    { schema_version: 2, type: 'message', text: 'future private input' },
    { schema_version: 1, type: 'message', text: 12 },
  ];
  incompatible.forEach((record, index) =>
    fs.writeFileSync(
      path.join(input, `held-${index}.json`),
      JSON.stringify(record),
    ),
  );
  const queue = new GroupQueue();
  let finish!: () => void;
  queue.setProcessMessagesFn(async () => {
    await new Promise<void>((resolve) => {
      finish = resolve;
    });
    return true;
  });
  queue.enqueueMessageCheck('tg:fixture');
  queue.registerProcess(
    'tg:fixture',
    new ChildProcess(),
    'private-fixture',
    'main',
  );
  try {
    expect(queue.sendMessage('tg:fixture', 'host follow-up input')).toBe(true);
    const files = fs.readdirSync(hostInput);
    expect(files).toHaveLength(1);
    expect(
      JSON.parse(fs.readFileSync(path.join(hostInput, files[0]), 'utf8')),
    ).toEqual({
      schema_version: 1,
      type: 'message',
      text: 'host follow-up input',
    });
    // Synthetic group-bind transport; preserve the actual host-produced bytes.
    fs.renameSync(path.join(hostInput, files[0]), path.join(input, files[0]));
    const child = run(
      'index.ts',
      JSON.stringify({
        prompt: 'initial',
        chatJid: 'tg:fixture',
        groupFolder: 'main',
        isMain: true,
      }),
      'drain',
    );
    expect(child.error).toBeUndefined();
    expect(child.status, child.stderr).toBe(0);
    const frames = [
      ...child.stdout.matchAll(
        /---NANOCLAW_OUTPUT_START---\s*([\s\S]*?)\s*---NANOCLAW_OUTPUT_END---/g,
      ),
    ].map((match) => JSON.parse(match[1]));
    expect(
      frames.find((frame) => frame.completion === 'sdk_turn_completed')?.result,
    ).toBe('initial\nhost follow-up input');
    incompatible.forEach((record, index) =>
      expect(
        fs.readFileSync(path.join(input, `held-${index}.json`), 'utf8'),
      ).toBe(JSON.stringify(record)),
    );
    expect(fs.existsSync(path.join(input, files[0]))).toBe(false);
  } finally {
    finish();
    await Promise.resolve();
  }
});

it.each(['{"wakeAgent":true}', '{"wakeAgent":true,"data":{}}'])(
  'actual precheck accepts object activation data or defaults omitted data: %s',
  (output) => {
    const child = run(
      'index.ts',
      JSON.stringify({
        prompt: 'fixed',
        chatJid: 'tg:fixture',
        groupFolder: 'main',
        isMain: true,
        isScheduledTask: true,
        script: 'printf "%s\\n" ' + "'" + output + "'",
      }),
      'drain',
    );
    expect(child.error).toBeUndefined();
    expect(child.status, child.stderr).toBe(0);
    expect(child.stdout).toContain('sdk_turn_completed');
    expect(child.stdout).not.toContain(
      'Task script did not return a valid precheck result',
    );
  },
);

it('actual malformed agent input flushes its complete frame before native exit', () => {
  const child = run('index.ts', '{broken', 'throw');
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
      error:
        'Failed to parse agent input; provide a valid container input record.',
    },
  ]);
});

it('actual host outer boundary flushes delayed stderr before exit without starting native services', () => {
  const hook = path.join(directory, 'host-hook.mjs');
  fs.writeFileSync(
    hook,
    `import {registerHooks} from 'node:module';
    const write=process.stderr.write.bind(process.stderr);process.stderr.write=(...args)=>{setTimeout(()=>write(...args),50);return true;};
    registerHooks({resolve(specifier,context,next){
      if(context.parentURL?.endsWith('/src/index.ts') && specifier==='./container-runtime.js')return {url:'data:text/javascript,'+encodeURIComponent('export function ensureContainerRuntimeRunning(){throw new Error("fixed synthetic startup refusal");}export function cleanupOrphans(){throw new Error("must not execute cleanup");}'),shortCircuit:true};
      return next(specifier,context);
    }});`,
  );
  const child = spawnSync(
    process.execPath,
    [
      '--import',
      fileURLToPath(import.meta.resolve('tsx/esm')),
      '--import',
      hook,
      fileURLToPath(new URL('./index.ts', import.meta.url)),
    ],
    {
      cwd: directory,
      env: { PATH: process.env.PATH ?? '', HOME: directory },
      encoding: 'utf8',
      timeout: 5000,
      maxBuffer: 262144,
    },
  );
  expect(child.error).toBeUndefined();
  expect(child.status).toBe(1);
  expect(child.stderr).toContain(
    'NanoClaw runtime failed; inspect the error before restarting',
  );
  expect(child.stderr).toContain('fixed synthetic startup refusal');
  expect(fs.readdirSync(directory)).toEqual(['host-hook.mjs']);
});

it('actual MCP producer refuses empty required strings before any durable publication', () => {
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
    ...[
      ['send_message', { text: '' }],
      ['react_to_message', { emoji: '' }],
      [
        'schedule_task',
        { prompt: '', schedule_type: 'interval', schedule_value: '5000' },
      ],
      ['pause_task', { task_id: '' }],
      ['resume_task', { task_id: '' }],
      ['cancel_task', { task_id: '' }],
      ['update_task', { task_id: '', prompt: 'fixed' }],
      ...['jid', 'name', 'folder', 'trigger'].map((field) => [
        'register_group',
        {
          jid: 'tg:fixed',
          name: 'fixed',
          folder: 'telegram_fixed',
          trigger: '@Andy',
          [field]: '',
        },
      ]),
    ].map(([name, args], index) => ({
      jsonrpc: '2.0',
      id: index + 2,
      method: 'tools/call',
      params: { name, arguments: args },
    })),
  ];
  const child = run(
    'ipc-mcp-stdio.ts',
    requests.map((row) => JSON.stringify(row)).join('\n') + '\n',
  );
  expect(child.error).toBeUndefined();
  expect(child.status, child.stderr).toBe(0);
  const replies = child.stdout
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  for (let id = 2; id < 13; id++)
    expect(replies.find((row) => row.id === id)?.result?.isError).toBe(true);
  expect(fs.existsSync(path.join(directory, 'workspace', 'ipc'))).toBe(false);
});

it('actual update tool rejects an omitted change but preserves explicit empty-script removal', () => {
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
      params: { name: 'update_task', arguments: { task_id: 'fixed-task' } },
    },
    {
      jsonrpc: '2.0',
      id: 3,
      method: 'tools/call',
      params: {
        name: 'update_task',
        arguments: { task_id: 'fixed-task', script: '' },
      },
    },
  ];
  const child = run(
    'ipc-mcp-stdio.ts',
    requests.map((row) => JSON.stringify(row)).join('\n') + '\n',
  );
  expect(child.error).toBeUndefined();
  expect(child.status, child.stderr).toBe(0);
  const replies = child.stdout
    .trim()
    .split('\n')
    .map((line) => JSON.parse(line));
  expect(replies.find((row) => row.id === 2)?.result?.isError).toBe(true);
  expect(replies.find((row) => row.id === 3)?.result?.isError ?? false).toBe(
    false,
  );
  const tasks = path.join(directory, 'workspace', 'ipc', 'tasks'),
    files = fs.readdirSync(tasks);
  expect(files).toHaveLength(1);
  expect(
    JSON.parse(fs.readFileSync(path.join(tasks, files[0]), 'utf8')),
  ).toMatchObject({
    schema_version: 1,
    type: 'update_task',
    taskId: 'fixed-task',
    script: '',
  });
});
