#!/usr/bin/env tsx
/**
 * One-off, idempotent helper for tweaking a registered group's
 * configuration in `messages.db`. Two modes (combinable in the same
 * run):
 *
 * 1. Append trigger keywords to `trigger_pattern.patterns` via
 *    `--add-keyword <pattern>` (repeatable). Existing
 *    (pattern, kind) entries are skipped with a log line.
 *
 * 2. Set `containerConfig.stage2Enabled` via
 *    `--set-stage2-enabled true|false`. Idempotent: a no-op when the
 *    column already holds the requested value.
 *
 * Generic enough to be reused for any group. Immediate drivers:
 *   - wtf chat: `bots` / `боты` keywords from the #83 batch run.
 *   - wtf chat: turn the Haiku classifier on (#83 follow-up).
 *
 * READ-WRITE on `messages.db`'s `registered_groups` row. Calls
 * `initDatabase()` so column defaults / migrations are in place
 * before we touch any column.
 *
 * Usage:
 *   tsx scripts/update-trigger-patterns.ts \
 *     --group <jid> \
 *     [--add-keyword <pattern> ...] \
 *     [--source <universal|learned|owner-set>] \
 *     [--set-stage2-enabled <true|false>] \
 *     [--dry-run]
 *
 * Examples:
 *   # add keywords to wtf
 *   tsx scripts/update-trigger-patterns.ts \
 *     --group tg:-1003869886477 \
 *     --add-keyword bots --add-keyword боты --source owner-set
 *
 *   # turn on stage 2 for wtf
 *   tsx scripts/update-trigger-patterns.ts \
 *     --group tg:-1003869886477 \
 *     --set-stage2-enabled true
 */
import path from 'path';
import { pathToFileURL } from 'url';

import {
  initDatabase,
  getRegisteredGroup,
  getTriggerPatterns,
  setRegisteredGroup,
  setTriggerPatterns,
} from '../src/db.js';
import type {
  TriggerPattern,
  TriggerPatternConfig,
  TriggerPatternSource,
} from '../src/types.js';

interface Args {
  group: string;
  addKeywords: string[];
  source: TriggerPatternSource;
  setStage2Enabled?: boolean;
  dryRun: boolean;
}

function parseArgs(argv: string[]): Args {
  const out: Partial<Args> & { addKeywords: string[] } = {
    addKeywords: [],
    source: 'owner-set',
    dryRun: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const a = argv[i];
    switch (a) {
      case '--group':
        out.group = argv[++i];
        break;
      case '--add-keyword':
        out.addKeywords.push(argv[++i]);
        break;
      case '--source': {
        const v = argv[++i];
        if (v !== 'owner-set' && v !== 'learned' && v !== 'universal') {
          throw new Error(
            `--source must be one of owner-set | learned | universal (got "${v}")`,
          );
        }
        out.source = v;
        break;
      }
      case '--set-stage2-enabled': {
        const v = argv[++i];
        if (v !== 'true' && v !== 'false') {
          throw new Error(
            `--set-stage2-enabled must be "true" or "false" (got "${v}")`,
          );
        }
        out.setStage2Enabled = v === 'true';
        break;
      }
      case '--dry-run':
        out.dryRun = true;
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${a}`);
    }
  }
  if (!out.group) throw new Error('--group <jid> is required');
  if (out.addKeywords.length === 0 && out.setStage2Enabled === undefined) {
    throw new Error(
      'must pass at least one of --add-keyword <pattern> or --set-stage2-enabled <true|false>',
    );
  }
  return out as Args;
}

function printHelp(): void {
  process.stdout.write(
    [
      'Usage: tsx scripts/update-trigger-patterns.ts [options]',
      '',
      'Required:',
      '  --group <jid>             Group JID (e.g. tg:-1003869886477)',
      '',
      'At least one of:',
      '  --add-keyword <pattern>   Keyword pattern (repeatable)',
      '  --set-stage2-enabled <true|false>',
      '                            Toggle containerConfig.stage2Enabled.',
      '',
      'Optional:',
      '  --source <kind>           owner-set (default) | learned | universal',
      '  --dry-run                 Print proposed change(s) and exit; no write.',
      '',
    ].join('\n'),
  );
}

function patternExists(
  cfg: TriggerPatternConfig,
  pattern: string,
  kind: TriggerPattern['kind'],
): boolean {
  return cfg.patterns.some((p) => p.pattern === pattern && p.kind === kind);
}

function makeKeywordPattern(
  pattern: string,
  source: TriggerPatternSource,
): TriggerPattern {
  return {
    pattern,
    kind: 'keyword',
    source,
    precision: 0,
    sample_count: 0,
    last_matched_at: null,
    last_updated_at: new Date().toISOString(),
  };
}

function main(): void {
  const args = parseArgs(process.argv);

  // initDatabase opens the prod messages.db at STORE_DIR/messages.db
  // and runs any pending migrations. Both setTriggerPatterns and
  // setRegisteredGroup assume the group already exists.
  initDatabase();

  // -------- Trigger-pattern updates --------

  let added: TriggerPattern[] = [];
  let next: TriggerPatternConfig | null = null;
  let beforeCount = 0;
  let triggerPlannedWrite = false;

  if (args.addKeywords.length > 0) {
    const before = getTriggerPatterns(args.group);
    if (before === undefined) {
      throw new Error(
        `No registered_groups row for jid "${args.group}". ` +
          `Register the group via the orchestrator before adding patterns.`,
      );
    }
    if (before === null) {
      throw new Error(
        `trigger_pattern column for "${args.group}" is unreadable ` +
          `(forward-version JSON or corrupt). Refusing to write.`,
      );
    }

    beforeCount = before.patterns.length;
    next = { version: 1, patterns: [...before.patterns] };
    const skipped: Array<{ pattern: string; kind: 'keyword' }> = [];

    for (const pattern of args.addKeywords) {
      if (patternExists(next, pattern, 'keyword')) {
        skipped.push({ pattern, kind: 'keyword' });
        process.stderr.write(
          `skip: keyword "${pattern}" already present for ${args.group}\n`,
        );
        continue;
      }
      const entry = makeKeywordPattern(pattern, args.source);
      next.patterns.push(entry);
      added.push(entry);
      process.stderr.write(
        `add: keyword "${pattern}" (source=${args.source}) → ${args.group}\n`,
      );
    }

    process.stderr.write(
      `\n--- trigger-pattern summary for ${args.group} ---\n` +
        `patterns before: ${beforeCount}\n` +
        `patterns after:  ${next.patterns.length}\n` +
        `added:           ${added.length}\n` +
        `skipped:         ${skipped.length}\n`,
    );

    if (added.length > 0) triggerPlannedWrite = true;
  }

  // -------- containerConfig.stage2Enabled update --------

  let stage2PlannedWrite = false;
  let stage2Before: boolean | undefined;
  let stage2After: boolean | undefined;
  if (args.setStage2Enabled !== undefined) {
    const row = getRegisteredGroup(args.group);
    if (!row) {
      throw new Error(
        `No registered_groups row for jid "${args.group}". ` +
          `Register the group via the orchestrator before toggling stage2Enabled.`,
      );
    }
    stage2Before = row.containerConfig?.stage2Enabled;
    stage2After = args.setStage2Enabled;
    process.stderr.write(
      `\n--- stage2Enabled for ${args.group} ---\n` +
        `before: ${JSON.stringify(stage2Before)}\n` +
        `after:  ${JSON.stringify(stage2After)}\n`,
    );
    if (stage2Before === stage2After) {
      process.stderr.write(
        `skip: stage2Enabled already at ${stage2After} for ${args.group}\n`,
      );
    } else {
      stage2PlannedWrite = true;
    }
  }

  if (!triggerPlannedWrite && !stage2PlannedWrite) {
    process.stderr.write('\nnothing to do — exiting without write.\n');
    return;
  }

  if (args.dryRun) {
    process.stderr.write('\n--- dry run: proposed change(s) ---\n');
    if (triggerPlannedWrite && next) {
      process.stderr.write('trigger_pattern config:\n');
      process.stdout.write(JSON.stringify(next, null, 2) + '\n');
    }
    if (stage2PlannedWrite) {
      process.stderr.write(
        `containerConfig.stage2Enabled: ${JSON.stringify(stage2Before)} → ${JSON.stringify(stage2After)}\n`,
      );
    }
    process.stderr.write('dry run — no write performed.\n');
    return;
  }

  if (triggerPlannedWrite && next) {
    setTriggerPatterns(args.group, next);
    const verify = getTriggerPatterns(args.group);
    if (!verify) {
      throw new Error(
        `verification readback failed for "${args.group}" after trigger write`,
      );
    }
    process.stderr.write(
      `wrote ${added.length} new pattern(s); verified count = ${verify.patterns.length}\n`,
    );
    for (const a of added) {
      const present = verify.patterns.some(
        (p) => p.pattern === a.pattern && p.kind === a.kind,
      );
      if (!present) {
        throw new Error(
          `verification: pattern "${a.pattern}" (${a.kind}) not present after write`,
        );
      }
    }
  }

  if (stage2PlannedWrite) {
    const row = getRegisteredGroup(args.group);
    if (!row) {
      throw new Error(
        `pre-write readback for stage2Enabled lost the row for "${args.group}"`,
      );
    }
    const nextConfig = {
      ...(row.containerConfig ?? {}),
      stage2Enabled: stage2After,
    };
    setRegisteredGroup(args.group, { ...row, containerConfig: nextConfig });
    const verify = getRegisteredGroup(args.group);
    if (!verify) {
      throw new Error(
        `verification readback failed for "${args.group}" after stage2 write`,
      );
    }
    if (verify.containerConfig?.stage2Enabled !== stage2After) {
      throw new Error(
        `verification: stage2Enabled = ${JSON.stringify(verify.containerConfig?.stage2Enabled)} after write (expected ${JSON.stringify(stage2After)})`,
      );
    }
    process.stderr.write(
      `wrote stage2Enabled = ${stage2After}; verified.\n`,
    );
  }

  process.stderr.write('ok.\n');
}

// ESM entry-point guard per `jbaruch/coding-policy: file-hygiene`.
if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  // Outer-boundary process contract per `coding-policy: error-handling`:
  // entry-point catch formats a single-line FATAL diagnostic for the
  // operator with a full stack still attached, then exits non-zero.
  // No recovery — the next statement is `process.exit(1)` — so the
  // bare catch here doesn't hide defects, it just produces nicer
  // stderr than the default unhandled-rejection trace.
  try {
    main();
  } catch (err: unknown) {
    const e = err instanceof Error ? err : new Error(String(err));
    process.stderr.write(`FATAL: ${e.message}\n${e.stack ?? ''}\n`);
    process.exit(1);
  }
}
