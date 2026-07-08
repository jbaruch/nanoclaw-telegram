#!/usr/bin/env tsx
/**
 * CLI wrapper around `src/db-dump.ts` (#398, #397). Dumps SQLite
 * state tables to `<outDir>/<table>.sql` for ad-hoc operator use;
 * the runtime path goes through `syncBackupRepo` in
 * `src/backup-sync.ts` instead.
 *
 * Usage:
 *   tsx scripts/dump-state-tables.ts --db-path store/messages.db \
 *     --out-dir groups/<group>/backup-repo/state
 *
 * Skipped tables (members of `STATE_TABLES` not present in this DB)
 * are reported in the JSON output — typical when a group hasn't run
 * a particular state-NNN migration yet.
 */
import path from 'path';
import { pathToFileURL } from 'url';

import { runDumpPlan } from '../src/db-dump.js';

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  const args = process.argv.slice(2);
  const flag = (name: string): string | undefined => {
    const idx = args.indexOf(name);
    return idx >= 0 ? args[idx + 1] : undefined;
  };
  const dbPath = flag('--db-path');
  const outDir = flag('--out-dir');
  const tablesArg = flag('--tables');
  if (!dbPath || !outDir) {
    process.stderr.write(
      'usage: tsx scripts/dump-state-tables.ts --db-path <db.sqlite> --out-dir <dir> [--tables a,b,c]\n',
    );
    process.exit(2);
  }
  const tables = tablesArg
    ? tablesArg
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
    : undefined;
  const report = runDumpPlan({ dbPath, outDir, tables });
  process.stdout.write(JSON.stringify(report, null, 2) + '\n');
}
