/**
 * Host-side coercion for the `scheduled_tasks.prompt` /
 * `scheduled_tasks.script` columns. Mirrors `coerceTaskPrompt` from
 * `container/agent-runner/src/format-task-row.ts` (separate TS project,
 * separate Docker image — the helper is duplicated rather than shared).
 *
 * #512 — without coercion at the IPC boundary, a non-string `prompt`
 * here lands as a SQLite BLOB via better-sqlite3, and `list_tasks`
 * later throws `t.prompt.slice is not a function`. A naive
 * `String(...)` coerce isn't enough either: it turns the
 * `JSON.stringify(Buffer)` shape (`{type:'Buffer',data:[...]}`) into
 * the literal string `"[object Object]"`, which silently persists a
 * meaningless prompt that fires with garbage at the next scheduled
 * tick. The contract this helper enforces is:
 *
 *   - `string` → passthrough (the normal case)
 *   - `{type:'Buffer',data:number[]}` → decoded UTF-8 via Buffer.from
 *   - anything else (object, number, array, …) → null, signalling the
 *     caller to reject the IPC payload rather than persist garbage
 *
 * `null`/`undefined` are caller-handled before this is reached (the
 * `data.prompt && …` gate in the schedule_task / update_task handlers
 * already drops falsy prompts up-front), so this helper doesn't carry
 * a "blank passthrough" branch — null in means caller-error in.
 */
export function coerceTaskTextField(value: unknown): string | null {
  if (typeof value === 'string') return value;
  if (
    typeof value === 'object' &&
    value !== null &&
    'type' in value &&
    (value as { type: unknown }).type === 'Buffer' &&
    'data' in value &&
    Array.isArray((value as { data: unknown }).data)
  ) {
    return Buffer.from((value as { data: number[] }).data).toString('utf8');
  }
  return null;
}
