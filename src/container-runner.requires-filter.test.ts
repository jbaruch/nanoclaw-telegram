// #552 — rule `requires:` frontmatter filter. Verifies the
// install-into-container loop in `buildVolumeMounts` filters rules
// whose declared gating skill is absent from the spawn's effective
// skill set, regardless of session class.
//
// Standalone test file (no global `vi.mock('fs')`) so the install
// loop runs against a real tmp registry. The narrow config mock
// points DATA_DIR / GROUPS_DIR / TILE_OWNER at tmp paths so the
// per-spawn `.tessl/` and `skills/` dirs land somewhere we can
// read back. Mirrors `container-runner.maintenance-blocklist.test.ts`
// — same fixture helpers, same import-after-mocks pattern.
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';

const TMP_PREFIX = 'nc-requires-';
let tmpRoot: string;
let registryRoot: string;
let groupsDir: string;
let dataDir: string;
let storeDir: string;

let ruleBlocklist: Set<string>;
let skillBlocklist: Set<string>;

vi.mock('./config.js', () => ({
  AGENT_AUTO_COMPACT_WINDOW: 800000,
  CONTAINER_IMAGE: 'nanoclaw-agent:latest',
  CONTAINER_MAX_OUTPUT_SIZE: 10485760,
  CONTAINER_TIMEOUT: 1800000,
  CREDENTIAL_PROXY_PORT: 3001,
  get DATA_DIR() {
    return dataDir;
  },
  ENABLE_THRESHOLD_NUKE: false,
  get GROUPS_DIR() {
    return groupsDir;
  },
  get STORE_DIR() {
    return storeDir;
  },
  HOST_PROJECT_ROOT: process.cwd(),
  HOST_UID: undefined,
  HOST_GID: undefined,
  IDLE_TIMEOUT: 1800000,
  get MAINTENANCE_RULE_BLOCKLIST() {
    return ruleBlocklist;
  },
  get MAINTENANCE_SKILL_BLOCKLIST() {
    return skillBlocklist;
  },
  MODEL_CONTEXT_WINDOW: 1000000,
  TILE_OWNER: 'test',
  TIMEZONE: 'UTC',
}));

vi.mock('better-sqlite3', () => ({ default: vi.fn() }));

const loggerCalls: Array<{ payload: unknown; msg: string }> = [];
vi.mock('./logger.js', () => ({
  logger: {
    debug: vi.fn(),
    info: vi.fn((payload: unknown, msg: string) =>
      loggerCalls.push({ payload, msg }),
    ),
    warn: vi.fn(),
    error: vi.fn(),
  },
}));

vi.mock('./host-logs.js', () => ({
  containerLogPath: vi.fn(() => '/dev/null'),
  ensureHostLogDirs: vi.fn(() => false),
  hostLogsDir: vi.fn(() => '/dev/null'),
  stripAnsi: (s: string) => s,
}));

vi.mock('./observer.js', () => ({ onAgentLine: vi.fn() }));

vi.mock('./credential-proxy.js', () => ({
  detectAuthMode: vi.fn(() => 'none'),
}));

vi.mock('./handoff.js', () => ({ isHandoffActive: vi.fn(() => false) }));

vi.mock('./ipc-input-sweep.js', () => ({ sweepStaleInputs: vi.fn() }));

vi.mock('./mount-security.js', () => ({
  validateAdditionalMounts: vi.fn(() => []),
}));

vi.mock('./env.js', () => ({ readEnvFile: vi.fn(() => ({})) }));

async function importSUT() {
  const mod = await import('./container-runner.js');
  return mod;
}

function writeFakeTile(
  tileName: string,
  rules: Record<string, string>,
  skills: Record<string, Record<string, string>>,
) {
  const tileRoot = path.join(registryRoot, 'tiles', 'test', tileName);
  const rulesDir = path.join(tileRoot, 'rules');
  fs.mkdirSync(rulesDir, { recursive: true });
  for (const [name, content] of Object.entries(rules)) {
    fs.writeFileSync(path.join(rulesDir, name), content);
  }
  for (const [skillName, files] of Object.entries(skills)) {
    const skillDir = path.join(tileRoot, 'skills', skillName);
    fs.mkdirSync(skillDir, { recursive: true });
    for (const [fname, fcontent] of Object.entries(files)) {
      const dst = path.join(skillDir, fname);
      fs.mkdirSync(path.dirname(dst), { recursive: true });
      fs.writeFileSync(dst, fcontent);
    }
  }
}

function makeGroup(folder: string) {
  fs.mkdirSync(path.join(groupsDir, folder), { recursive: true });
  return {
    name: folder,
    folder,
    trigger: '@bot',
    added_at: new Date().toISOString(),
    containerConfig: { trusted: false },
    requiresTrigger: true,
    isMain: false,
  };
}

function jidFor(folder: string): string {
  return `tg:test-${folder}`;
}

let originalCwd: string;

describe('#552 rule requires: filter', () => {
  beforeEach(() => {
    originalCwd = process.cwd();
    tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), TMP_PREFIX));
    registryRoot = path.join(tmpRoot, 'tessl-workspace', '.tessl');
    groupsDir = path.join(tmpRoot, 'groups');
    dataDir = path.join(tmpRoot, 'data');
    storeDir = path.join(tmpRoot, 'store');
    fs.mkdirSync(registryRoot, { recursive: true });
    fs.mkdirSync(groupsDir, { recursive: true });
    fs.mkdirSync(dataDir, { recursive: true });
    fs.mkdirSync(storeDir, { recursive: true });
    process.chdir(tmpRoot);
    ruleBlocklist = new Set();
    skillBlocklist = new Set();
    loggerCalls.length = 0;
  });

  afterEach(() => {
    process.chdir(originalCwd);
    fs.rmSync(tmpRoot, { recursive: true, force: true });
    vi.resetModules();
  });

  function rulesDstFor(folder: string, sessionName: string, tileName: string) {
    return path.join(
      dataDir,
      'sessions',
      folder,
      sessionName,
      '.claude',
      '.tessl',
      'tiles',
      'test',
      tileName,
      'rules',
    );
  }

  function aggregatedRulesMdFor(folder: string, sessionName: string) {
    return path.join(
      dataDir,
      'sessions',
      folder,
      sessionName,
      '.claude',
      '.tessl',
      'RULES.md',
    );
  }

  it('rule without requires: loads unconditionally (regression guard for non-#552 rules)', async () => {
    // The vast majority of existing rules carry no `requires:`
    // field. Their behavior must be unchanged — they always land
    // in both the per-tile mirror and the aggregated RULES.md.
    writeFakeTile(
      'nanoclaw-core',
      { 'plain-rule.md': '---\nalwaysApply: true\n---\n\n# Plain\n' },
      {
        // One skill so the skill-presence pre-scan has something to
        // walk — proves the rule loads regardless of what's there.
        'some-skill': { 'SKILL.md': 'name: some-skill\n' },
      },
    );
    const { buildVolumeMounts } = await importSUT();
    const group = makeGroup('test-plain');
    buildVolumeMounts(group, false, jidFor(group.folder), 'default');

    expect(
      fs.existsSync(
        path.join(
          rulesDstFor(group.folder, 'default', 'nanoclaw-core'),
          'plain-rule.md',
        ),
      ),
    ).toBe(true);
    expect(
      fs.readFileSync(aggregatedRulesMdFor(group.folder, 'default'), 'utf8'),
    ).toContain('# Plain');
  });

  it('rule with requires: [present-skill] loads when the skill is in the spawn', async () => {
    writeFakeTile(
      'nanoclaw-core',
      {
        'gated-rule.md':
          '---\nalwaysApply: true\nrequires: [present-skill]\n---\n\n# Gated\n',
      },
      { 'present-skill': { 'SKILL.md': 'name: present-skill\n' } },
    );
    const { buildVolumeMounts } = await importSUT();
    const group = makeGroup('test-present');
    buildVolumeMounts(group, false, jidFor(group.folder), 'default');

    expect(
      fs.existsSync(
        path.join(
          rulesDstFor(group.folder, 'default', 'nanoclaw-core'),
          'gated-rule.md',
        ),
      ),
    ).toBe(true);
    expect(
      fs.readFileSync(aggregatedRulesMdFor(group.folder, 'default'), 'utf8'),
    ).toContain('# Gated');
  });

  it('rule with requires: [absent-skill] is filtered out and reported in the install log', async () => {
    writeFakeTile(
      'nanoclaw-core',
      {
        'gated-rule.md':
          '---\nalwaysApply: true\nrequires: [absent-skill]\n---\n\n# Gated\n',
        'plain-rule.md': '---\nalwaysApply: true\n---\n\n# Plain\n',
      },
      { 'unrelated-skill': { 'SKILL.md': 'name: unrelated-skill\n' } },
    );
    const { buildVolumeMounts } = await importSUT();
    const group = makeGroup('test-absent');
    buildVolumeMounts(group, false, jidFor(group.folder), 'default');

    // Gated rule is omitted from both the per-tile mirror and the
    // aggregated RULES.md. The unrelated plain rule still lands.
    expect(
      fs.existsSync(
        path.join(
          rulesDstFor(group.folder, 'default', 'nanoclaw-core'),
          'gated-rule.md',
        ),
      ),
    ).toBe(false);
    expect(
      fs.existsSync(
        path.join(
          rulesDstFor(group.folder, 'default', 'nanoclaw-core'),
          'plain-rule.md',
        ),
      ),
    ).toBe(true);
    const rulesContent = fs.readFileSync(
      aggregatedRulesMdFor(group.folder, 'default'),
      'utf8',
    );
    expect(rulesContent).not.toContain('# Gated');
    expect(rulesContent).toContain('# Plain');

    const filterCalls = loggerCalls.filter(
      (c) => c.msg === 'install_blocklist_filtered',
    );
    expect(filterCalls).toHaveLength(1);
    const payload = filterCalls[0].payload as { filteredRules: string[] };
    expect(payload.filteredRules).toEqual([
      'nanoclaw-core/gated-rule.md (requires: absent-skill)',
    ]);
  });

  it('rule with requires: [absent, present] loads (any-of semantics)', async () => {
    writeFakeTile(
      'nanoclaw-core',
      {
        'any-of-rule.md':
          '---\nalwaysApply: true\nrequires: [absent-a, present-b, absent-c]\n---\n\n# AnyOf\n',
      },
      { 'present-b': { 'SKILL.md': 'name: present-b\n' } },
    );
    const { buildVolumeMounts } = await importSUT();
    const group = makeGroup('test-any-of');
    buildVolumeMounts(group, false, jidFor(group.folder), 'default');

    expect(
      fs.readFileSync(aggregatedRulesMdFor(group.folder, 'default'), 'utf8'),
    ).toContain('# AnyOf');
  });

  it('rule requiring a tile skill loads when that skill is in the same tile', async () => {
    // The most common shape: a rule and its gating skill ship in
    // the same tile. The skill pre-scan walks every installed tile
    // so the rule loads correctly.
    writeFakeTile(
      'nanoclaw-untrusted',
      {
        'no-orphan-tasks.md':
          '---\nalwaysApply: true\nrequires: [schedule-task]\n---\n\n# NoOrphan\n',
      },
      { 'schedule-task': { 'SKILL.md': 'name: schedule-task\n' } },
    );
    const { buildVolumeMounts } = await importSUT();
    const group = makeGroup('test-same-tile');
    buildVolumeMounts(group, false, jidFor(group.folder), 'default');

    expect(
      fs.readFileSync(aggregatedRulesMdFor(group.folder, 'default'), 'utf8'),
    ).toContain('# NoOrphan');
  });

  it('rule requiring a skill from ANOTHER tile loads when that tile is installed', async () => {
    // A rule in tile A can require a skill in tile B — the
    // skill-presence pre-scan walks every installed tile, so the
    // cross-tile gating relationship resolves. This is the shape
    // step 3 of the #552 plan uses to keep rules in their original
    // tile while gating them on skills that live elsewhere.
    writeFakeTile(
      'nanoclaw-core',
      {
        'cross-tile-rule.md':
          '---\nalwaysApply: true\nrequires: [other-tile-skill]\n---\n\n# CrossTile\n',
      },
      {},
    );
    writeFakeTile(
      'nanoclaw-untrusted',
      {},
      {
        'other-tile-skill': { 'SKILL.md': 'name: other-tile-skill\n' },
      },
    );
    const { buildVolumeMounts } = await importSUT();
    const group = makeGroup('test-cross-tile');
    buildVolumeMounts(group, false, jidFor(group.folder), 'default');

    expect(
      fs.readFileSync(aggregatedRulesMdFor(group.folder, 'default'), 'utf8'),
    ).toContain('# CrossTile');
  });

  it('rule requiring a built-in skill loads when the built-in is present', async () => {
    // Built-in skills (`wiki`, `agent-browser`, `status`) live
    // under `<cwd>/container/skills/`, not in any tile. The pre-
    // scan walks that dir too, so a rule can gate on a built-in.
    writeFakeTile(
      'nanoclaw-untrusted',
      {
        'wiki-rule.md':
          '---\nalwaysApply: true\nrequires: [wiki]\n---\n\n# WikiAware\n',
      },
      {},
    );
    const builtinWikiDir = path.join(tmpRoot, 'container', 'skills', 'wiki');
    fs.mkdirSync(builtinWikiDir, { recursive: true });
    fs.writeFileSync(
      path.join(builtinWikiDir, 'SKILL.md'),
      '# wiki\n\nThe wiki skill.\n',
    );

    const { buildVolumeMounts } = await importSUT();
    const group = makeGroup('test-builtin');
    buildVolumeMounts(group, false, jidFor(group.folder), 'default');

    expect(
      fs.readFileSync(aggregatedRulesMdFor(group.folder, 'default'), 'utf8'),
    ).toContain('# WikiAware');
  });

  it('rule requiring a maintenance-blocklisted skill is filtered (closure also drops the rule)', async () => {
    // Combined #337 + #552 case: maintenance class blocks a skill,
    // so its prompt isn't loaded — and a rule that requires it
    // should also be filtered (its content has nothing to gate
    // against). The blocklist and the requires-filter agree.
    skillBlocklist = new Set(['blocked-skill']);
    writeFakeTile(
      'nanoclaw-core',
      {
        'gated-rule.md':
          '---\nalwaysApply: true\nrequires: [blocked-skill]\n---\n\n# Gated\n',
      },
      { 'blocked-skill': { 'SKILL.md': 'name: blocked-skill\n' } },
    );
    const { buildVolumeMounts } = await importSUT();
    const group = makeGroup('test-maint-blocked');
    buildVolumeMounts(group, false, jidFor(group.folder), 'maintenance');

    expect(
      fs.existsSync(
        path.join(
          rulesDstFor(group.folder, 'maintenance', 'nanoclaw-core'),
          'gated-rule.md',
        ),
      ),
    ).toBe(false);
  });

  it('rule requiring a skill exempted via #544 closure DOES load (closure feeds reachable set)', async () => {
    // The reachable set the rule filter consults is the SAME one
    // the #544 closure computes — a skill blocklisted but rescued
    // via `Skill()` reference from a loaded root is in `reachable`,
    // and a rule requiring it must therefore load. This is the
    // load-bearing integration between #544 and #552.
    skillBlocklist = new Set(['wiki']);
    writeFakeTile(
      'nanoclaw-core',
      {
        'wiki-rule.md':
          '---\nalwaysApply: true\nrequires: [wiki]\n---\n\n# WikiRule\n',
      },
      {
        'wiki-lint': {
          'SKILL.md': '# wiki-lint\n\n`Skill(skill: "wiki")`\n',
        },
      },
    );
    const builtinWikiDir = path.join(tmpRoot, 'container', 'skills', 'wiki');
    fs.mkdirSync(builtinWikiDir, { recursive: true });
    fs.writeFileSync(
      path.join(builtinWikiDir, 'SKILL.md'),
      '# wiki\n\nBuilt-in.\n',
    );

    const { buildVolumeMounts } = await importSUT();
    const group = makeGroup('test-closure-rescue');
    buildVolumeMounts(group, false, jidFor(group.folder), 'maintenance');

    expect(
      fs.readFileSync(
        aggregatedRulesMdFor(group.folder, 'maintenance'),
        'utf8',
      ),
    ).toContain('# WikiRule');
  });

  it('rule with requires: [] never loads (author explicitly opted out)', async () => {
    // `requires: []` is the author saying "no skill satisfies this
    // — the rule is parked but never loads". Distinct from a
    // missing declaration (which loads unconditionally).
    writeFakeTile(
      'nanoclaw-core',
      {
        'parked-rule.md':
          '---\nalwaysApply: true\nrequires: []\n---\n\n# Parked\n',
      },
      { 'any-skill': { 'SKILL.md': 'name: any-skill\n' } },
    );
    const { buildVolumeMounts } = await importSUT();
    const group = makeGroup('test-parked');
    buildVolumeMounts(group, false, jidFor(group.folder), 'default');

    expect(
      fs.existsSync(
        path.join(
          rulesDstFor(group.folder, 'default', 'nanoclaw-core'),
          'parked-rule.md',
        ),
      ),
    ).toBe(false);
    const rulesContent = fs.existsSync(
      aggregatedRulesMdFor(group.folder, 'default'),
    )
      ? fs.readFileSync(aggregatedRulesMdFor(group.folder, 'default'), 'utf8')
      : '';
    expect(rulesContent).not.toContain('# Parked');
  });
});
