import fs from 'fs';
import readline from 'readline';

/**
 * Parser for the SDK's session JSONL transcripts.
 *
 * Path: `data/sessions/<group>/<slot>/.claude/projects/<slug>/<sessionId>.jsonl`
 *
 * Each line is one of: `assistant`, `user`, `system`, `result`. Tool
 * invocations live inside assistant messages as content blocks with
 * `type: 'tool_use'`; their results land inside subsequent user
 * messages as `type: 'tool_result'` blocks referencing the
 * `tool_use_id`. The parser walks the file once, builds the tool_use
 * → tool_result pairing, and yields completed `ToolInvocation`
 * records in execution order.
 *
 * Used by the kill-auto-compaction `## Facts` writer (design doc §2,
 * `docs/proposals/kill-auto-compaction.md`). The writer filters
 * invocations through `classifyTool()` to keep only the
 * state-mutating ones in the "do NOT re-execute" list.
 */

export interface ToolInvocation {
  /** Tool name as the SDK emitted it (e.g. `Write`, `Bash`,
   *  `mcp__nanoclaw__send_message`). */
  name: string;
  /** SDK-assigned id for the tool call. Lets the writer dedupe across
   *  multiple JSONL passes if the file is appended-to between reads. */
  id: string;
  /** Arguments the SDK invoked the tool with. Untyped because the
   *  shape varies per tool — the checkpoint writer summarises the
   *  most-load-bearing keys per tool family rather than dumping the
   *  whole object. */
  input: Record<string, unknown>;
  /** ISO-8601 timestamp the tool_use entry was written to the JSONL. */
  startedAt: string;
  /** ISO-8601 timestamp the corresponding tool_result was written.
   *  `null` when the JSONL ended before the result landed (in-flight
   *  invocation at threshold-cross time). */
  completedAt: string | null;
  /** Whether the tool_result reported success. `null` when the result
   *  hasn't landed yet. */
  isError: boolean | null;
}

interface JsonlLine {
  type?: string;
  timestamp?: string;
  message?: {
    content?: Array<{
      type?: string;
      id?: string;
      name?: string;
      input?: Record<string, unknown>;
      tool_use_id?: string;
      is_error?: boolean;
    }>;
  };
}

/**
 * Parse a session JSONL file, yielding completed and in-flight tool
 * invocations in execution order.
 *
 * Streams the file line-by-line via readline so a 100-MB transcript
 * doesn't pin 100 MB of RSS. Lines that fail JSON.parse are skipped
 * silently — the SDK's transcript format is generally well-formed,
 * but a torn write at the tail (orchestrator killed mid-flush)
 * shouldn't make the parser explode.
 */
export async function parseSessionTranscript(
  jsonlPath: string,
): Promise<ToolInvocation[]> {
  if (!fs.existsSync(jsonlPath)) return [];

  const stream = fs.createReadStream(jsonlPath, { encoding: 'utf-8' });
  const rl = readline.createInterface({ input: stream, crlfDelay: Infinity });

  const invocations = new Map<string, ToolInvocation>();
  const order: string[] = [];

  for await (const line of rl) {
    if (!line.trim()) continue;
    let parsed: JsonlLine;
    try {
      parsed = JSON.parse(line) as JsonlLine;
    } catch (err) {
      // Only swallow SyntaxError — that's the expected shape for a
      // torn write at tail or other malformed JSON line. Anything
      // else (TypeError, RangeError, etc.) indicates a bug in the
      // parser itself and must propagate. The design doc's
      // failure-mode table (§7) says a corrupt checkpoint is
      // treated as missing; the same posture applies here, but
      // narrowed to the exception type the JSON parser actually
      // throws on bad input.
      if (err instanceof SyntaxError) continue;
      throw err;
    }

    const ts = parsed.timestamp;
    const content = parsed.message?.content;
    if (!Array.isArray(content)) continue;

    for (const block of content) {
      if (block.type === 'tool_use' && block.id && block.name) {
        invocations.set(block.id, {
          name: block.name,
          id: block.id,
          input: block.input ?? {},
          startedAt: ts ?? '',
          completedAt: null,
          isError: null,
        });
        order.push(block.id);
      } else if (block.type === 'tool_result' && block.tool_use_id) {
        const inv = invocations.get(block.tool_use_id);
        if (inv) {
          inv.completedAt = ts ?? null;
          inv.isError = block.is_error === true;
        }
      }
    }
  }

  return order.map((id) => invocations.get(id)!).filter(Boolean);
}
