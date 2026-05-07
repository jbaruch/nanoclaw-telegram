/**
 * Tests for the pre-shutdown checkpoint writer (#497).
 *
 * Pins:
 *   - Active default sessions get a checkpoint file written before
 *     queue.shutdown / deploy.sh force-kill discards them.
 *   - Maintenance-slot containers and inactive default slots are
 *     skipped (only mid-turn user-facing work needs reentry context).
 *   - One write failure does not block the others (best-effort, mirrors
 *     the threshold-cross handler's per-group try/catch).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import fs from 'fs';
import os from 'os';
import path from 'path';

let tmpRoot: string;
let groupsDir: string;
let dataDir: string;

vi.mock('./config.js', async () => {
  const actual =
    await vi.importActual<typeof import('./config.js')>('./config.js');
  return {
    ...actual,
    get GROUPS_DIR() {
      return groupsDir;
    },
    get DATA_DIR() {
      return dataDir;
    },
  };
});

import { computeThresholds } from './threshold.js';
import { writeShutdownCheckpoints } from './shutdown-checkpoints.js';
import { checkpointPaths } from './checkpoint.js';

const DEFAULT = 'default';
const MAINTENANCE = 'maintenance';

const mkLogger = () => ({
  info: vi.fn(),
  error: vi.fn(),
});

const mkThresholds = () => computeThresholds(200_000);

beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nc-shutdown-ckpt-'));
  groupsDir = path.join(tmpRoot, 'groups');
  dataDir = path.join(tmpRoot, 'data');
  fs.mkdirSync(groupsDir, { recursive: true });
  fs.mkdirSync(dataDir, { recursive: true });
});

afterEach(() => {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

describe('writeShutdownCheckpoints', () => {
  it('writes a checkpoint for an active default-slot session', async () => {
    const folder = 'telegram_alpha';
    fs.mkdirSync(path.join(groupsDir, folder), { recursive: true });
    const logger = mkLogger();
    const written = await writeShutdownCheckpoints({
      active: [
        { groupJid: 'jid-1', sessionName: DEFAULT, groupFolder: folder },
      ],
      sessions: { [folder]: { [DEFAULT]: 'session-aaa' } },
      registeredGroups: { 'jid-1': { name: 'Alpha' } },
      thresholds: mkThresholds(),
      defaultSessionName: DEFAULT,
      dataDir,
      logger,
    });
    expect(written).toBe(1);
    const cp = checkpointPaths(path.join(groupsDir, folder));
    expect(fs.existsSync(cp.live)).toBe(true);
    const body = fs.readFileSync(cp.live, 'utf8');
    // The checkpoint body uses the group's display name, not the folder
    // — verifies the registeredGroups lookup wired through.
    expect(body).toContain('Alpha');
    // Trigger framing is shutdown, not threshold-cross — pins the
    // operator-facing distinction during incident debugging.
    expect(body).toContain('graceful shutdown');
    expect(body).toContain('**Trigger:** shutdown');
    expect(body).toContain('**Tokens used:** unknown');
    expect(body).not.toContain('**Tokens used at trigger:** 0');
    expect(body).toContain('force-killed on shutdown');
    expect(logger.info).toHaveBeenCalledWith(
      expect.objectContaining({ group: folder, sessionId: 'session-aaa' }),
      'Wrote pre-shutdown checkpoint',
    );
  });

  it('falls back to folder when registered name is empty string (truthy fallback)', async () => {
    const folder = 'telegram_emptyname';
    fs.mkdirSync(path.join(groupsDir, folder), { recursive: true });
    const logger = mkLogger();
    await writeShutdownCheckpoints({
      active: [
        { groupJid: 'jid-1', sessionName: DEFAULT, groupFolder: folder },
      ],
      sessions: { [folder]: { [DEFAULT]: 'session-empty-name' } },
      registeredGroups: { 'jid-1': { name: '' } },
      thresholds: mkThresholds(),
      defaultSessionName: DEFAULT,
      dataDir,
      logger,
    });
    const body = fs.readFileSync(
      checkpointPaths(path.join(groupsDir, folder)).live,
      'utf8',
    );
    // The Group: line MUST contain the folder name, not be a blank.
    expect(body).toContain(`**Group:** ${folder}`);
  });

  it('skips maintenance-slot containers (only default slot needs reentry)', async () => {
    const folder = 'telegram_beta';
    fs.mkdirSync(path.join(groupsDir, folder), { recursive: true });
    const logger = mkLogger();
    const written = await writeShutdownCheckpoints({
      active: [
        { groupJid: 'jid-1', sessionName: MAINTENANCE, groupFolder: folder },
      ],
      sessions: { [folder]: { [MAINTENANCE]: 'session-zzz' } },
      registeredGroups: { 'jid-1': { name: 'Beta' } },
      thresholds: mkThresholds(),
      defaultSessionName: DEFAULT,
      dataDir,
      logger,
    });
    expect(written).toBe(0);
    const cp = checkpointPaths(path.join(groupsDir, folder));
    expect(fs.existsSync(cp.live)).toBe(false);
  });

  it('skips containers without a tracked sessionId', async () => {
    const folder = 'telegram_gamma';
    fs.mkdirSync(path.join(groupsDir, folder), { recursive: true });
    const logger = mkLogger();
    const written = await writeShutdownCheckpoints({
      active: [
        { groupJid: 'jid-1', sessionName: DEFAULT, groupFolder: folder },
      ],
      sessions: {}, // no entry → no sessionId
      registeredGroups: { 'jid-1': { name: 'Gamma' } },
      thresholds: mkThresholds(),
      defaultSessionName: DEFAULT,
      dataDir,
      logger,
    });
    expect(written).toBe(0);
    expect(logger.error).not.toHaveBeenCalled();
  });

  it('continues across groups when one write fails', async () => {
    // First group has no folder on disk — resolveGroupFolderPath
    // succeeds (validates the name pattern only) but the write inside
    // writeCheckpoint will fail because the parent dir is missing and
    // .checkpoints can't be created. Second group has its folder
    // provisioned and should still get a checkpoint.
    //
    // Wait — actually resolveGroupFolderPath only validates the name,
    // and writeCheckpoint creates `<groupDir>/.checkpoints` if absent
    // via fs.mkdirSync(..., { recursive: true }). To force a real
    // failure, we use an invalid folder name on the first entry so
    // assertValidGroupFolder throws.
    const goodFolder = 'telegram_good';
    fs.mkdirSync(path.join(groupsDir, goodFolder), { recursive: true });
    const logger = mkLogger();
    const written = await writeShutdownCheckpoints({
      active: [
        { groupJid: 'jid-bad', sessionName: DEFAULT, groupFolder: '../bad' },
        { groupJid: 'jid-good', sessionName: DEFAULT, groupFolder: goodFolder },
      ],
      sessions: {
        '../bad': { [DEFAULT]: 'session-bad' },
        [goodFolder]: { [DEFAULT]: 'session-good' },
      },
      registeredGroups: {
        'jid-bad': { name: 'Bad' },
        'jid-good': { name: 'Good' },
      },
      thresholds: mkThresholds(),
      defaultSessionName: DEFAULT,
      dataDir,
      logger,
    });
    expect(written).toBe(1);
    expect(logger.error).toHaveBeenCalledWith(
      expect.objectContaining({ group: '../bad' }),
      'Pre-shutdown checkpoint write failed',
    );
    const cp = checkpointPaths(path.join(groupsDir, goodFolder));
    expect(fs.existsSync(cp.live)).toBe(true);
  });

  it('falls back to the folder name when registeredGroups lookup misses', async () => {
    const folder = 'telegram_unregistered';
    fs.mkdirSync(path.join(groupsDir, folder), { recursive: true });
    const logger = mkLogger();
    const written = await writeShutdownCheckpoints({
      active: [
        { groupJid: 'unknown-jid', sessionName: DEFAULT, groupFolder: folder },
      ],
      sessions: { [folder]: { [DEFAULT]: 'session-orphan' } },
      registeredGroups: {}, // empty — no matching jid
      thresholds: mkThresholds(),
      defaultSessionName: DEFAULT,
      dataDir,
      logger,
    });
    expect(written).toBe(1);
    const cp = checkpointPaths(path.join(groupsDir, folder));
    expect(fs.existsSync(cp.live)).toBe(true);
  });
});
