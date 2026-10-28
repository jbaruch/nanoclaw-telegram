/** Reader of host-owned current_tasks.json schema1; see docs/TASK-SNAPSHOT.md.
 * Metadata is a hint only. Every mutation is revalidated against the host DB.
 */
import fs from 'fs';
export class TaskSnapshotError extends Error {}
export interface TaskSnapshotRow {
  id: string;
  groupFolder: string;
  prompt: string;
  script: string | null;
  schedule_type: string;
  schedule_value: string;
  status: string;
  next_run: string | null;
}
const fields = [
  'id',
  'groupFolder',
  'prompt',
  'script',
  'schedule_type',
  'schedule_value',
  'status',
  'next_run',
]
  .sort()
  .join();
export function readTaskSnapshot(file: string): TaskSnapshotRow[] {
  const fd = fs.openSync(
    file,
    fs.constants.O_RDONLY | fs.constants.O_NOFOLLOW | fs.constants.O_NONBLOCK,
  );
  using _lease = { [Symbol.dispose]: () => fs.closeSync(fd) };
  const maximum = 4 * 1024 * 1024,
    stat = fs.fstatSync(fd);
  if (!stat.isFile() || stat.nlink !== 1 || stat.size > maximum)
    throw new TaskSnapshotError(
      'Task snapshot is unavailable; refresh it from the host before use.',
    );
  const buffer = Buffer.alloc(maximum + 1);
  let size = 0;
  while (size < buffer.length) {
    const count = fs.readSync(fd, buffer, size, buffer.length - size, null);
    if (!count) break;
    size += count;
  }
  if (size > maximum)
    throw new TaskSnapshotError(
      'Task snapshot exceeds its reader budget; use complete update fields or repair host snapshot coverage.',
    );
  let parsed: unknown;
  try {
    parsed = JSON.parse(
      new TextDecoder('utf-8', { fatal: true }).decode(
        buffer.subarray(0, size),
      ),
    );
  } catch (error) {
    if (
      !(error instanceof SyntaxError) &&
      !(
        error instanceof TypeError &&
        'code' in error &&
        error.code === 'ERR_ENCODING_INVALID_ENCODED_DATA'
      )
    )
      throw error;
    throw new TaskSnapshotError(
      'Task snapshot encoding/JSON is invalid; refresh it from the host.',
      { cause: error },
    );
  }
  if (
    !parsed ||
    typeof parsed !== 'object' ||
    Array.isArray(parsed) ||
    Object.keys(parsed).sort().join() !== 'schema_version,tasks' ||
    !('schema_version' in parsed) ||
    parsed.schema_version !== 1 ||
    !('tasks' in parsed) ||
    !Array.isArray(parsed.tasks)
  )
    throw new TaskSnapshotError(
      'Task snapshot schema is unsupported; refresh the co-shipped host and agent before use.',
    );
  const ids = new Set<string>(),
    rows: TaskSnapshotRow[] = [];
  for (const value of parsed.tasks) {
    if (
      !value ||
      typeof value !== 'object' ||
      Array.isArray(value) ||
      Object.keys(value).sort().join() !== fields
    )
      throw new TaskSnapshotError(
        'Task snapshot row is malformed; refresh it from the host.',
      );
    const row = value as Record<string, unknown>;
    if (
      ![
        'id',
        'groupFolder',
        'prompt',
        'schedule_type',
        'schedule_value',
        'status',
      ].every((key) => typeof row[key] === 'string') ||
      !(typeof row.id === 'string' && row.id) ||
      !(typeof row.groupFolder === 'string' && row.groupFolder) ||
      (row.script !== null && typeof row.script !== 'string') ||
      (row.next_run !== null && typeof row.next_run !== 'string') ||
      ids.has(row.id)
    )
      throw new TaskSnapshotError(
        'Task snapshot rows are malformed or ambiguous; refresh it from the host.',
      );
    ids.add(row.id);
    rows.push(row as unknown as TaskSnapshotRow);
  }
  return rows;
}
