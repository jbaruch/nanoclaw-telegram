#!/usr/bin/env tsx
/**
 * CLI wrapper around `src/audit-precheck-gating.ts` (#375). Emits the
 * snapshot JSON the audit-replay agent diffs against
 * `docs/precheck-gating-audit.md`.
 *
 * Usage:
 *   tsx scripts/audit-precheck-gating.ts --db store/messages.db
 *   tsx scripts/audit-precheck-gating.ts --days 30
 */
import path from 'path';
import { pathToFileURL } from 'url';

import { runAuditSnapshot } from '../src/audit-precheck-gating.js';

const ALLOWED_FLAGS = new Set(['--db', '--days']);

function parseArgs(argv: string[]): { dbArg?: string; daysArg?: string } {
  // Strict positional parser: every token must be one of the allowed
  // flags, immediately followed by a non-flag value. Pre-fix, the
  // scan-and-pluck `args.indexOf('--db')` resolved `--db --days 90`
  // to dbPath="--days" and silently dropped unknown flags. Reject
  // any anomaly with exit 2 so a typo is loud, not data-corrupting.
  const out: { dbArg?: string; daysArg?: string } = {};
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (!ALLOWED_FLAGS.has(token)) {
      throw new Error(`unknown argument '${token}'`);
    }
    const value = argv[i + 1];
    if (value === undefined || ALLOWED_FLAGS.has(value)) {
      throw new Error(`flag ${token} requires a value`);
    }
    if (token === '--db') {
      if (out.dbArg !== undefined) throw new Error(`duplicate --db`);
      out.dbArg = value;
    } else {
      if (out.daysArg !== undefined) throw new Error(`duplicate --days`);
      out.daysArg = value;
    }
    i++;
  }
  return out;
}

function parsePositiveInt(s: string): number {
  // `Number.parseInt('1.5', 10)` silently returns 1; `Number()`
  // returns 1.5 then `Number.isInteger` rejects it. Match the
  // operator's --days promise: positive integer or exit 2.
  if (!/^\d+$/.test(s)) {
    throw new Error(`'${s}' is not a positive integer`);
  }
  const n = Number(s);
  if (!Number.isInteger(n) || n <= 0) {
    throw new Error(`'${s}' is not a positive integer`);
  }
  return n;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  let dbArg: string | undefined;
  let daysArg: string | undefined;
  try {
    ({ dbArg, daysArg } = parseArgs(process.argv.slice(2)));
  } catch (e) {
    if (!(e instanceof Error)) throw e;
    process.stderr.write(
      `audit-precheck-gating: ${e.message}\nusage: tsx scripts/audit-precheck-gating.ts [--db <path>] [--days <int>]\n`,
    );
    process.exit(2);
  }

  const dbPath = dbArg
    ? path.resolve(dbArg)
    : path.resolve(process.cwd(), 'store/messages.db');

  let days = 90;
  if (daysArg !== undefined) {
    try {
      days = parsePositiveInt(daysArg);
    } catch (e) {
      if (!(e instanceof Error)) throw e;
      process.stderr.write(`audit-precheck-gating: --days ${e.message}\n`);
      process.exit(2);
    }
  }

  const snap = runAuditSnapshot({ dbPath, windowDays: days });
  process.stdout.write(JSON.stringify(snap, null, 2) + '\n');
}
