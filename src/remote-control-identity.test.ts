import { spawn, type ChildProcess } from 'child_process';
import fs from 'fs';
import { once } from 'events';
import { afterEach, expect, it, vi } from 'vitest';
import {
  captureRemoteControlIdentity,
  parseRemoteControlIdentity,
  verifyRemoteControlIdentity,
} from './remote-control-identity.js';

const nonce = '00000000-0000-4000-8000-000000000001';
const children: ChildProcess[] = [];
async function nativeChild(hide = false): Promise<ChildProcess> {
  const child = spawn(
    process.execPath,
    [
      '-e',
      `${hide ? "process.title='fixed-hidden-command';" : ''}process.send({ready:true});setInterval(()=>{},60000);`,
      '--',
      `NanoClaw Remote ${nonce}`,
    ],
    { stdio: ['ignore', 'ignore', 'ignore', 'ipc'], detached: true, env: {} },
  );
  children.push(child);
  let deadline: ReturnType<typeof setTimeout>;
  try {
    const ready = await Promise.race([
      once(child, 'message'),
      new Promise<never>((_resolve, reject) => {
        deadline = setTimeout(() => {
          child.kill('SIGKILL');
          reject(
            new Error(
              'Owned native identity fixture did not signal readiness.',
            ),
          );
        }, 5000);
      }),
    ]);
    expect(ready[0]).toEqual({ ready: true });
    return child;
  } finally {
    clearTimeout(deadline!);
  }
}
afterEach(async () => {
  vi.restoreAllMocks();
  const pending = children.splice(0).map(async (child) => {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const ended = once(child, 'exit');
    child.kill('SIGKILL');
    await ended;
  });
  await Promise.all(pending);
});

it('binds a genuine owned native child, never a caller boolean or a different command record', async () => {
  const child = await nativeChild();
  const identity = captureRemoteControlIdentity(child.pid!, nonce);
  expect(parseRemoteControlIdentity(identity)).toEqual(identity);
  expect(verifyRemoteControlIdentity(child.pid!, identity)).toBe('same');
  expect(
    verifyRemoteControlIdentity(child.pid!, {
      ...identity,
      command_sha256: '0'.repeat(64),
    }),
  ).toBe('changed');
  const ended = once(child, 'exit');
  child.kill('SIGKILL');
  await ended;
  expect(verifyRemoteControlIdentity(child.pid!, identity)).toBe('absent');
});

it('does not turn a hidden native argv into portable nonce confidence', async () => {
  const child = await nativeChild(true);
  if (process.platform === 'darwin')
    expect(() => captureRemoteControlIdentity(child.pid!, nonce)).toThrow(
      'does not expose the owned instance marker',
    );
  else
    expect(captureRemoteControlIdentity(child.pid!, nonce).kind).toBe(
      'linux-birth',
    );
});

it.skipIf(process.platform !== 'linux')(
  'refuses a recalled native birth from another boot or start tick without a signal',
  async () => {
    const child = await nativeChild();
    const identity = captureRemoteControlIdentity(child.pid!, nonce);
    expect(identity.kind).toBe('linux-birth');
    if (identity.kind !== 'linux-birth')
      throw new Error('Linux birth proof required.');
    const signal = vi.spyOn(process, 'kill');
    expect(
      verifyRemoteControlIdentity(child.pid!, {
        ...identity,
        start_ticks: (BigInt(identity.start_ticks) + 1n).toString(),
      }),
    ).toBe('changed');
    expect(
      verifyRemoteControlIdentity(child.pid!, { ...identity, boot_id: nonce }),
    ).toBe('changed');
    expect(signal).not.toHaveBeenCalled();
  },
);

it.skipIf(process.platform !== 'linux')(
  'refuses inconsistent native executable observations',
  async () => {
    const child = await nativeChild(),
      readlink = fs.readlinkSync.bind(fs);
    let reads = 0;
    vi.spyOn(fs, 'readlinkSync').mockImplementation((file, options) => {
      if (String(file) === `/proc/${child.pid}/exe` && ++reads === 2)
        return Buffer.from('/changed-native-executable') as never;
      return readlink(file, options as never);
    });
    expect(() => captureRemoteControlIdentity(child.pid!, nonce)).toThrow(
      'changed during observation',
    );
  },
);

it.each([
  null,
  true,
  {},
  { kind: 'future', command_sha256: 'a'.repeat(64) },
  {
    kind: 'darwin-command-nonce',
    nonce,
    executable_name: 'node',
    command_sha256: 'a'.repeat(64),
    extra: true,
  },
])('refuses malformed or unknown stored identity %j', (value) => {
  expect(() => parseRemoteControlIdentity(value)).toThrow();
});
