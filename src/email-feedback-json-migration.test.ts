import Database from 'better-sqlite3';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { describe, it, expect, vi } from 'vitest';

// Each scenario builds an isolated tempDir + chdir so the
// CWD-rooted `STORE_DIR` and `GROUPS_DIR` constants in `src/config.ts`
// resolve under it. Mirrors the pattern in
// `src/morning-brief-pending-json-migration.test.ts`.

interface FeedbackRow {
  pattern?: unknown;
  label?: unknown;
  source?: unknown;
  date?: unknown;
}

function writeEmailFeedbackFile(
  tempDir: string,
  groupName: string,
  payload: { feedback: FeedbackRow[] } | FeedbackRow[] | unknown,
): string {
  const folder = path.join(tempDir, 'groups', groupName);
  fs.mkdirSync(folder, { recursive: true });
  const filePath = path.join(folder, 'email-feedback.json');
  fs.writeFileSync(filePath, JSON.stringify(payload, null, 2));
  return filePath;
}

async function runWithTempDir<T>(
  fn: (tempDir: string) => Promise<T>,
): Promise<T> {
  const repoRoot = process.cwd();
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-efb-'));
  try {
    process.chdir(tempDir);
    fs.mkdirSync(path.join(tempDir, 'store'), { recursive: true });
    return await fn(tempDir);
  } finally {
    process.chdir(repoRoot);
    fs.rmSync(tempDir, { recursive: true, force: true });
  }
}

describe('email-feedback.json → SQLite migration (#295)', () => {
  it('imports feedback rows, assigns AUTOINCREMENT ids, and renames the source file', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeEmailFeedbackFile(tempDir, 'telegram_main', {
        feedback: [
          {
            pattern: 'noreply@shop.example.com',
            label: 'noise',
            source: 'baruch-response',
            date: '2026-04-15',
          },
          {
            pattern: 'invoice from acme',
            label: 'actionable',
            source: 'baruch-response',
            date: '2026-04-16',
          },
          {
            pattern: 'newsletter unsubscribe',
            label: 'noise',
            source: 'baruch-response',
            date: '2026-04-17',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const rows = db
            .prepare(
              'SELECT id, pattern, label, source, date FROM email_feedback ORDER BY id',
            )
            .all() as Array<{
            id: number;
            pattern: string;
            label: string;
            source: string;
            date: string;
          }>;
          expect(rows).toHaveLength(3);
          // AUTOINCREMENT-assigned ids — we never supply them at
          // INSERT time, the DB picks them. Three sequential rows in
          // a fresh DB → ids 1, 2, 3.
          expect(rows.map((r) => r.id)).toEqual([1, 2, 3]);
          expect(rows[0]).toMatchObject({
            pattern: 'noreply@shop.example.com',
            label: 'noise',
            source: 'baruch-response',
            date: '2026-04-15',
          });
          expect(rows[1]).toMatchObject({
            pattern: 'invoice from acme',
            label: 'actionable',
          });
          expect(rows[2]).toMatchObject({
            pattern: 'newsletter unsubscribe',
            label: 'noise',
          });
        } finally {
          db.close();
        }

        // Source renamed to .migrated-<YYYY-MM-DD>.
        expect(fs.existsSync(filePath)).toBe(false);
        const renamed = fs
          .readdirSync(path.dirname(filePath))
          .filter((f) => f.startsWith('email-feedback.json.migrated-'));
        expect(renamed).toHaveLength(1);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('accepts a bare-array payload as if it were the wrapped shape', async () => {
    // Issue #295's body documents the legacy shape as a bare array.
    // The wrapped `{feedback: [...]}` shape is preferred, but the
    // migration also accepts bare arrays so legacy files in the
    // wild still import cleanly.
    await runWithTempDir(async (tempDir) => {
      const filePath = writeEmailFeedbackFile(tempDir, 'telegram_main', [
        {
          pattern: 'legacy-pattern',
          label: 'noise',
          source: 'baruch-response',
          date: '2026-04-10',
        },
      ]);

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const rows = db
            .prepare('SELECT pattern, label, source FROM email_feedback')
            .all() as Array<Record<string, unknown>>;
          expect(rows).toHaveLength(1);
          expect(rows[0]).toMatchObject({
            pattern: 'legacy-pattern',
            label: 'noise',
            source: 'baruch-response',
          });
        } finally {
          db.close();
        }
        expect(fs.existsSync(filePath)).toBe(false);
      } finally {
        _closeDatabase();
      }
    });
  });

  it("falls back to schema 'baruch-response' default when the JSON-era row omits `source`", async () => {
    // The state-002 schema has `source TEXT NOT NULL DEFAULT
    // 'baruch-response'`. The migration omits the column from the
    // INSERT when the row doesn't carry it so the DDL default fires
    // — same shape as the morning-brief migration's `added` /
    // CURRENT_TIMESTAMP fallback. Lock that contract here.
    await runWithTempDir(async (tempDir) => {
      writeEmailFeedbackFile(tempDir, 'telegram_main', {
        feedback: [
          {
            pattern: 'no-source-row',
            label: 'actionable',
            // source omitted on purpose
            date: '2026-04-18',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const row = db.prepare('SELECT source FROM email_feedback').get() as {
            source: string;
          };
          expect(row.source).toBe('baruch-response');
        } finally {
          db.close();
        }
      } finally {
        _closeDatabase();
      }
    });
  });

  it('renames the source file and leaves the table empty when the feedback array is empty', async () => {
    await runWithTempDir(async (tempDir) => {
      const filePath = writeEmailFeedbackFile(tempDir, 'telegram_main', {
        feedback: [],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      initDatabase();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const count = (
            db.prepare('SELECT COUNT(*) AS n FROM email_feedback').get() as {
              n: number;
            }
          ).n;
          expect(count).toBe(0);
        } finally {
          db.close();
        }
        // Empty array is still a valid array — file gets renamed so
        // the next boot doesn't keep re-reading it.
        expect(fs.existsSync(filePath)).toBe(false);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('rolls back and leaves the source in place when a row violates the label CHECK constraint', async () => {
    // The schema's `CHECK(label IN ('actionable', 'noise'))` rejects
    // any other value. The catch-narrowing helper turns the
    // SqliteError SQLITE_CONSTRAINT_CHECK into a warn, the
    // transaction rolls back, and the source file stays put for
    // human triage. Lock that contract here.
    await runWithTempDir(async (tempDir) => {
      const goodFile = writeEmailFeedbackFile(tempDir, 'good', {
        feedback: [
          {
            pattern: 'good-row',
            label: 'noise',
            source: 'baruch-response',
            date: '2026-04-15',
          },
        ],
      });
      const badFile = writeEmailFeedbackFile(tempDir, 'bad', {
        feedback: [
          {
            pattern: 'bad-row',
            label: 'noisy', // violates the CHECK
            source: 'baruch-response',
            date: '2026-04-15',
          },
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // Bad file's transaction rolled back: zero rows for the bad
          // pattern.
          const badCount = (
            db
              .prepare(
                'SELECT COUNT(*) AS n FROM email_feedback WHERE pattern = ?',
              )
              .get('bad-row') as { n: number }
          ).n;
          expect(badCount).toBe(0);
          // Good file still imported.
          const goodCount = (
            db
              .prepare(
                'SELECT COUNT(*) AS n FROM email_feedback WHERE pattern = ?',
              )
              .get('good-row') as { n: number }
          ).n;
          expect(goodCount).toBe(1);
        } finally {
          db.close();
        }
        // Good file renamed; bad file left in place.
        expect(fs.existsSync(goodFile)).toBe(false);
        expect(fs.existsSync(badFile)).toBe(true);
        // Per-file warn fired with the constraint-violation message
        // and the SQLite error code.
        const constraintWarnFired = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(msg && msg.includes('violated a DB constraint'));
        });
        expect(constraintWarnFired).toBe(true);
        const errCodeIsCheck = warnSpy.mock.calls.some((call) => {
          const meta = call.find(
            (arg) =>
              typeof arg === 'object' &&
              arg !== null &&
              'errCode' in (arg as object),
          ) as { errCode?: string } | undefined;
          return Boolean(
            meta?.errCode && meta.errCode.startsWith('SQLITE_CONSTRAINT_'),
          );
        });
        expect(errCodeIsCheck).toBe(true);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('skips a row missing required fields (pattern / label / date) with a warning', async () => {
    // The schema enforces NOT NULL on pattern, label, and date. The
    // migration pre-guards each row so a single malformed entry
    // doesn't roll back the whole per-file transaction — skip with
    // a warn instead.
    await runWithTempDir(async (tempDir) => {
      writeEmailFeedbackFile(tempDir, 'telegram_main', {
        feedback: [
          {
            pattern: 'good-pattern',
            label: 'actionable',
            source: 'baruch-response',
            date: '2026-04-15',
          },
          {
            // pattern omitted
            label: 'noise',
            source: 'baruch-response',
            date: '2026-04-15',
          },
          {
            pattern: 'no-label',
            // label omitted
            source: 'baruch-response',
            date: '2026-04-15',
          },
          {
            pattern: 'no-date',
            label: 'noise',
            source: 'baruch-response',
            // date omitted
          },
          // Non-object element — separately covered by the
          // isObjectRow guard.
          'not an object',
          null,
        ],
      });

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          // Only the well-formed row landed.
          const rows = db
            .prepare('SELECT pattern FROM email_feedback')
            .all() as Array<{ pattern: string }>;
          expect(rows.map((r) => r.pattern)).toEqual(['good-pattern']);
        } finally {
          db.close();
        }
        // Warn fired for each missing-required row plus the
        // non-object rows.
        const missingFieldWarns = warnSpy.mock.calls.filter((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(msg && msg.includes('missing required fields'));
        });
        expect(missingFieldWarns).toHaveLength(3);
        const nonObjectWarns = warnSpy.mock.calls.filter((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(msg && msg.includes('non-object row'));
        });
        expect(nonObjectWarns).toHaveLength(2);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('is a no-op when the source file has already been renamed (re-run)', async () => {
    // Idempotency contract: once the source is renamed to
    // .migrated-<YYYY-MM-DD>, the existsSync gate skips it. No
    // ON CONFLICT handling because the import is append-only.
    await runWithTempDir(async (tempDir) => {
      const folder = path.join(tempDir, 'groups', 'telegram_main');
      fs.mkdirSync(folder, { recursive: true });
      const renamedPath = path.join(
        folder,
        'email-feedback.json.migrated-2026-04-20',
      );
      fs.writeFileSync(
        renamedPath,
        JSON.stringify({
          feedback: [
            {
              pattern: 'already-imported',
              label: 'noise',
              source: 'baruch-response',
              date: '2026-04-15',
            },
          ],
        }),
      );

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      expect(() => initDatabase()).not.toThrow();
      try {
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const count = (
            db.prepare('SELECT COUNT(*) AS n FROM email_feedback').get() as {
              n: number;
            }
          ).n;
          expect(count).toBe(0);
        } finally {
          db.close();
        }
        // Renamed file untouched.
        expect(fs.existsSync(renamedPath)).toBe(true);
      } finally {
        _closeDatabase();
      }
    });
  });

  it('skips a malformed JSON file with a warning and leaves it in place', async () => {
    // Mirror the morning-brief and orders tests: one valid file plus
    // one malformed file. Malformed must NOT abort the pass — the
    // valid file still imports.
    await runWithTempDir(async (tempDir) => {
      const goodFile = writeEmailFeedbackFile(tempDir, 'good', {
        feedback: [
          {
            pattern: 'good-pattern',
            label: 'noise',
            source: 'baruch-response',
            date: '2026-04-15',
          },
        ],
      });
      const badFolder = path.join(tempDir, 'groups', 'bad');
      fs.mkdirSync(badFolder, { recursive: true });
      const badFile = path.join(badFolder, 'email-feedback.json');
      fs.writeFileSync(badFile, '{ this is not valid json');

      vi.resetModules();
      const { initDatabase, _closeDatabase } = await import('./db.js');
      const { logger } = await import('./logger.js');
      const warnSpy = vi.spyOn(logger, 'warn');
      try {
        initDatabase();
        const db = new Database(path.join(tempDir, 'store', 'messages.db'));
        try {
          const rows = db
            .prepare('SELECT pattern FROM email_feedback')
            .all() as Array<{ pattern: string }>;
          expect(rows.map((r) => r.pattern)).toEqual(['good-pattern']);
        } finally {
          db.close();
        }
        expect(fs.existsSync(goodFile)).toBe(false);
        expect(fs.existsSync(badFile)).toBe(true);
        const malformedWarnFired = warnSpy.mock.calls.some((call) => {
          const msg = call.find((arg) => typeof arg === 'string') as
            | string
            | undefined;
          return Boolean(msg && msg.includes('invalid JSON'));
        });
        expect(malformedWarnFired).toBe(true);
      } finally {
        warnSpy.mockRestore();
        _closeDatabase();
      }
    });
  });

  it('skips a non-object/non-array JSON payload (null / number / string) with a warning', async () => {
    // `JSON.parse` returns null / numbers / strings for syntactically
    // valid but non-object/non-array payloads. Bare arrays ARE
    // accepted (legacy shape), so the warn-and-skip applies only to
    // the other non-object cases.
    for (const payload of ['null', '42', '"hello"']) {
      await runWithTempDir(async (tempDir) => {
        const goodFile = writeEmailFeedbackFile(tempDir, 'good', {
          feedback: [
            {
              pattern: `good-${payload.replace(/\W/g, '')}`,
              label: 'noise',
              source: 'baruch-response',
              date: '2026-04-15',
            },
          ],
        });
        const badFolder = path.join(tempDir, 'groups', 'bad');
        fs.mkdirSync(badFolder, { recursive: true });
        const badFile = path.join(badFolder, 'email-feedback.json');
        fs.writeFileSync(badFile, payload);

        vi.resetModules();
        const { initDatabase, _closeDatabase } = await import('./db.js');
        const { logger } = await import('./logger.js');
        const warnSpy = vi.spyOn(logger, 'warn');
        try {
          expect(() => initDatabase()).not.toThrow();
          const db = new Database(path.join(tempDir, 'store', 'messages.db'));
          try {
            const rows = db
              .prepare('SELECT pattern FROM email_feedback')
              .all() as Array<{ pattern: string }>;
            expect(rows).toHaveLength(1);
          } finally {
            db.close();
          }
          expect(fs.existsSync(goodFile)).toBe(false);
          expect(fs.existsSync(badFile)).toBe(true);
          const nonObjectWarnFired = warnSpy.mock.calls.some((call) => {
            const msg = call.find((arg) => typeof arg === 'string') as
              | string
              | undefined;
            return Boolean(
              msg && msg.includes('payload is not an object or array'),
            );
          });
          expect(nonObjectWarnFired).toBe(true);
        } finally {
          warnSpy.mockRestore();
          _closeDatabase();
        }
      });
    }
  });
});
