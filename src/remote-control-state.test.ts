import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const fixture = vi.hoisted(() => ({ directory: '' }));
vi.mock('./config.js', () => ({
  get DATA_DIR() {
    return fixture.directory;
  },
}));
let owner: typeof import('./remote-control.js');
const legacy = {
  pid: 77777,
  url: 'https://claude.ai/code?bridge=env_fixed',
  startedBy: 'fixed',
  startedInChat: 'tg:123',
  startedAt: '2024-01-01T00:00:00Z',
};
beforeEach(async () => {
  fixture.directory = fs.mkdtempSync(
    path.join(os.tmpdir(), 'nanoclaw-remote-state-'),
  );
  vi.resetModules();
  owner = await import('./remote-control.js');
  vi.spyOn(process, 'kill').mockReturnValue(true);
});
afterEach(() => {
  vi.restoreAllMocks();
  fs.rmSync(fixture.directory, { recursive: true });
});

describe('atomic Remote Control owner migration', () => {
  it('preserves legacy bytes on failed replacement and can retry the migration', () => {
    const state = path.join(fixture.directory, 'remote-control.json');
    const original = JSON.stringify(legacy);
    fs.writeFileSync(state, original);
    const denied = Object.assign(new Error('fixed replacement denied'), {
      code: 'EACCES',
    });
    const rename = vi.spyOn(fs, 'renameSync').mockImplementationOnce(() => {
      throw denied;
    });
    expect(owner.restoreRemoteControl).toThrow(denied);
    expect(fs.readFileSync(state, 'utf8')).toBe(original);
    expect(fs.readdirSync(fixture.directory)).toEqual(['remote-control.json']);
    expect(owner.getActiveSession()).toBeNull();
    rename.mockRestore();
    owner.restoreRemoteControl();
    expect(JSON.parse(fs.readFileSync(state, 'utf8'))).toEqual({
      schema_version: 1,
      ...legacy,
    });
    expect(fs.statSync(state).mode & 0o777).toBe(0o600);
    expect(owner.getActiveSession()).toEqual(legacy);
  });

  it('holds unknown-version bytes without process or filesystem mutation', () => {
    const state = path.join(fixture.directory, 'remote-control.json');
    const raw = JSON.stringify({ schema_version: 2, ...legacy });
    fs.writeFileSync(state, raw);
    expect(owner.restoreRemoteControl).toThrow(
      'Unsupported Remote Control state version',
    );
    expect(fs.readFileSync(state, 'utf8')).toBe(raw);
    expect(fs.readdirSync(fixture.directory)).toEqual(['remote-control.json']);
    expect(process.kill).not.toHaveBeenCalled();
  });
});
