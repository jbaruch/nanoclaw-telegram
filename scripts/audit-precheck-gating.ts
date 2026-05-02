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

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  const args = process.argv.slice(2);
  const flag = (name: string): string | undefined => {
    const idx = args.indexOf(name);
    return idx >= 0 ? args[idx + 1] : undefined;
  };
  const dbArg = flag('--db');
  const daysArg = flag('--days');

  const dbPath = dbArg
    ? path.resolve(dbArg)
    : path.resolve(process.cwd(), 'store/messages.db');
  const days = daysArg ? Number.parseInt(daysArg, 10) : 90;
  if (!Number.isFinite(days) || days <= 0) {
    process.stderr.write(
      `audit-precheck-gating: --days must be a positive integer (got ${daysArg})\n`,
    );
    process.exit(2);
  }

  const snap = runAuditSnapshot({ dbPath, windowDays: days });
  process.stdout.write(JSON.stringify(snap, null, 2) + '\n');
}
