/**
 * Pure formatter for `list_tasks` rows. Lifted out of the
 * `ipc-mcp-stdio.ts` MCP-tool closure so it's directly testable and so
 * the prompt-shape coercion has one obvious home.
 *
 * #512 — `current_tasks.json` is produced by the host
 * (`writeTasksSnapshot` in `src/container-runner.ts`) by JSON.stringify-
 * ing rows from `getAllTasks()` straight off SQLite. better-sqlite3
 * returns a TEXT-typed column as `string` and a BLOB-typed column as
 * `Buffer`; JSON.stringify writes the Buffer as
 * `{"type":"Buffer","data":[...]}`. The prior formatter assumed
 * `t.prompt` was always `string` and called `.slice` directly, which
 * threw `t.prompt.slice is not a function` on the first BLOB row and
 * (because of the outer try/catch) hid the entire task list behind an
 * opaque error.
 *
 * This module accepts an `unknown` task and an `unknown` prompt,
 * coerces the prompt to a string defensively, and returns the
 * formatted line. The host-side fix in `src/ipc.ts` String-coerces at
 * the writer boundary so new rows can never land as BLOB; this reader-
 * side defense keeps `list_tasks` non-fatal against any historical row
 * that's still BLOB-typed (or any future writer that bypasses the
 * coerce).
 */

export interface RawTaskRow {
  id?: unknown;
  prompt?: unknown;
  schedule_type?: unknown;
  schedule_value?: unknown;
  status?: unknown;
  next_run?: unknown;
}

/**
 * Coerce a raw task-row prompt of unknown shape to a string.
 *
 * The shapes seen on the wire:
 *   - `string` (TEXT-typed column, the normal case)
 *   - `{ type: 'Buffer', data: number[] }` (BLOB-typed column, post
 *     `JSON.stringify(Buffer)`)
 *   - `null` / `undefined` (defensive — shouldn't happen, but the
 *     formatter must never throw)
 *   - Anything else — coerced via `String()` so the row still appears
 *     in the listing rather than poisoning the whole output
 */
export function coerceTaskPrompt(prompt: unknown): string {
  if (typeof prompt === 'string') return prompt;
  if (prompt == null) return '';
  if (
    typeof prompt === 'object' &&
    'type' in prompt &&
    (prompt as { type: unknown }).type === 'Buffer' &&
    'data' in prompt &&
    Array.isArray((prompt as { data: unknown }).data)
  ) {
    return Buffer.from((prompt as { data: number[] }).data).toString('utf8');
  }
  return String(prompt);
}

export function formatTaskRow(t: RawTaskRow): string {
  const id = typeof t.id === 'string' ? t.id : String(t.id ?? '');
  const promptStr = coerceTaskPrompt(t.prompt);
  const scheduleType =
    typeof t.schedule_type === 'string'
      ? t.schedule_type
      : String(t.schedule_type ?? '');
  const scheduleValue =
    typeof t.schedule_value === 'string'
      ? t.schedule_value
      : String(t.schedule_value ?? '');
  const status = typeof t.status === 'string' ? t.status : String(t.status ?? '');
  const nextRun =
    typeof t.next_run === 'string' && t.next_run.length > 0 ? t.next_run : 'N/A';
  return `- [${id}] ${promptStr.slice(0, 50)}... (${scheduleType}: ${scheduleValue}) - ${status}, next: ${nextRun}`;
}
