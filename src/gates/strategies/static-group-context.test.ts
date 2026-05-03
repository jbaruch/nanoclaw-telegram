// Stage 2 default-strategy tests. We use a tmpdir + a config.js mock
// so `resolveGroupFolderPath` writes into our test sandbox rather
// than the real `groups/` tree.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

let tmpRoot: string;
let groupsDir: string;
let dataDir: string;

// Owner identity is mutable per-test so we can exercise the
// suppression cases (one-set, both-unset). The mocked module reads
// these via getters so individual tests can flip values inside
// `beforeEach`.
let ownerName: string | undefined;
let ownerHandle: string | undefined;
// Mutable assistant-username fixture so tests can exercise both
// single-handle and multi-handle (#464) deployments without
// re-mounting the module.
let assistantUsernames: string[] = ['testbot'];

vi.mock('../../config.js', async () => {
  const actual =
    await vi.importActual<typeof import('../../config.js')>('../../config.js');
  return {
    ...actual,
    get GROUPS_DIR() {
      return groupsDir;
    },
    get DATA_DIR() {
      return dataDir;
    },
    ASSISTANT_NAME: 'TestAssistant',
    get ASSISTANT_USERNAME() {
      return assistantUsernames[0];
    },
    get ASSISTANT_USERNAMES() {
      return assistantUsernames;
    },
    get ASSISTANT_OWNER_NAME() {
      return ownerName;
    },
    get ASSISTANT_OWNER_HANDLE() {
      return ownerHandle;
    },
  };
});

import { _initTestDatabase, _writeRawRegisteredGroup } from '../../db.js';
import { staticGroupContextStrategy } from './static-group-context.js';
import type { GateContext } from '../index.js';

const TEST_FOLDER = 'telegram_strategytest';
const TEST_JID = 'strategytest@g.us';

function buildCtx(overrides: Partial<GateContext> = {}): GateContext {
  return {
    groupJid: TEST_JID,
    groupFolder: TEST_FOLDER,
    message: { text: 'hi', senderJid: 's@s.whatsapp.net' },
    triggerPatterns: null,
    ...overrides,
  };
}

beforeEach(() => {
  tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'nc-strategy-'));
  groupsDir = path.join(tmpRoot, 'groups');
  dataDir = path.join(tmpRoot, 'data');
  // Default to "owner not configured" for every test; cases that
  // exercise the owner block flip these locally before calling
  // buildContext.
  ownerName = undefined;
  ownerHandle = undefined;
  // Default to a single-handle install; multi-handle tests overwrite.
  assistantUsernames = ['testbot'];
  fs.mkdirSync(groupsDir, { recursive: true });
  fs.mkdirSync(dataDir, { recursive: true });
  _initTestDatabase();
  _writeRawRegisteredGroup({
    jid: TEST_JID,
    name: 'Strategy Test Group',
    folder: TEST_FOLDER,
    trigger: '@andy',
    added_at: '2024-01-01T00:00:00Z',
    container_config: null,
  });
});

afterEach(() => {
  fs.rmSync(tmpRoot, { recursive: true, force: true });
});

describe('staticGroupContextStrategy.buildContext', () => {
  it('emits group name + assistant identity + CLAUDE.md head when present', async () => {
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.mkdirSync(groupDir, { recursive: true });
    fs.writeFileSync(
      path.join(groupDir, 'CLAUDE.md'),
      '# Test Group\nThis is a test description.\nLine 3\n',
    );
    const out = await staticGroupContextStrategy.buildContext(buildCtx());
    expect(out).toContain('Strategy Test Group');
    expect(out).toContain('TestAssistant');
    expect(out).toContain('# Test Group');
    expect(out).toContain('This is a test description.');
    expect(out).toContain('Group CLAUDE.md (head):');
  });

  it('emits both the display name and the Telegram @-handle for the assistant', async () => {
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.mkdirSync(groupDir, { recursive: true });
    const out = await staticGroupContextStrategy.buildContext(buildCtx());
    // The "Assistant identity:" block is load-bearing: the classifier
    // prompt's input description points at it. Don't pin the exact
    // line shape — just assert presence of the section header and
    // both identity forms.
    expect(out).toContain('Assistant identity:');
    expect(out).toContain('TestAssistant');
    expect(out).toContain('@testbot');
  });

  // #464: multi-handle bots — when more than one Telegram handle
  // resolves to the same assistant (autocomplete-only handle plus
  // an internal/vocative one), the strategy must list every alias
  // so the classifier prompt's identity-match rule recognizes a
  // mention of any of them. The plural label keys the prompt's
  // alias-resolution rule.
  it('lists every alias when ASSISTANT_USERNAMES has multiple entries', async () => {
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.mkdirSync(groupDir, { recursive: true });
    assistantUsernames = ['testbot', 'testbotsurebot'];
    const out = await staticGroupContextStrategy.buildContext(buildCtx());
    expect(out).toContain('Assistant identity:');
    expect(out).toContain('@testbot');
    expect(out).toContain('@testbotsurebot');
    expect(out).toContain('aliases');
  });

  it('uses the singular handle label for single-handle installs', async () => {
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.mkdirSync(groupDir, { recursive: true });
    // Default fixture (set in beforeEach) is single-handle.
    const out = await staticGroupContextStrategy.buildContext(buildCtx());
    expect(out).toContain('Telegram @-handle: @testbot');
    expect(out).not.toContain('aliases');
  });

  it('returns minimal context when CLAUDE.md is missing (no throw)', async () => {
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.mkdirSync(groupDir, { recursive: true });
    const out = await staticGroupContextStrategy.buildContext(buildCtx());
    expect(out).toContain('Strategy Test Group');
    expect(out).toContain('TestAssistant');
    expect(out).toContain('Group CLAUDE.md: not present');
  });

  it('truncates to first 200 lines', async () => {
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.mkdirSync(groupDir, { recursive: true });
    const lines = Array.from({ length: 400 }, (_, i) => `line-${i}`);
    fs.writeFileSync(path.join(groupDir, 'CLAUDE.md'), lines.join('\n'));
    const out = await staticGroupContextStrategy.buildContext(buildCtx());
    expect(out).toContain('line-0');
    expect(out).toContain('line-199');
    expect(out).not.toContain('line-200');
  });

  // --- Owner identity (Stage 2 owner-aware classification) ---
  //
  // The classifier rule keys off the literal substring
  // `Owner: <Name> (@<handle>)` so it can compare against the inbound
  // message's `Sender:` field. These tests pin both the emit path
  // (both vars set) and the suppression paths (one missing, neither
  // set) since a half-rendered owner line would silently degrade the
  // classifier rule.

  it('emits Owner line when both ASSISTANT_OWNER_NAME and ASSISTANT_OWNER_HANDLE are set', async () => {
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.mkdirSync(groupDir, { recursive: true });
    ownerName = 'Test Owner';
    ownerHandle = 'testowner';
    const out = await staticGroupContextStrategy.buildContext(buildCtx());
    expect(out).toContain('Owner: Test Owner (@testowner)');
  });

  it('suppresses Owner line when only ASSISTANT_OWNER_NAME is set', async () => {
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.mkdirSync(groupDir, { recursive: true });
    ownerName = 'Test Owner';
    ownerHandle = undefined;
    const out = await staticGroupContextStrategy.buildContext(buildCtx());
    expect(out).not.toContain('Owner:');
  });

  it('suppresses Owner line when only ASSISTANT_OWNER_HANDLE is set', async () => {
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.mkdirSync(groupDir, { recursive: true });
    ownerName = undefined;
    ownerHandle = 'testowner';
    const out = await staticGroupContextStrategy.buildContext(buildCtx());
    expect(out).not.toContain('Owner:');
  });

  it('suppresses Owner line when neither owner var is set (default)', async () => {
    const groupDir = path.join(groupsDir, TEST_FOLDER);
    fs.mkdirSync(groupDir, { recursive: true });
    // ownerName / ownerHandle remain undefined from beforeEach.
    const out = await staticGroupContextStrategy.buildContext(buildCtx());
    expect(out).not.toContain('Owner:');
  });
});
