/**
 * #443 — Parser for the orchestrator's pretty-printed host log
 * (`data/host-logs/orchestrator.log`).
 *
 * The format is produced by `src/logger.ts:formatData` and is NOT
 * pino — each record renders as a header line followed by zero or
 * more indented field lines:
 *
 *     [HH:mm:ss.SSS] LEVEL (pid): msg-text
 *         field1: <JSON.stringify(value)>
 *         field2: <JSON.stringify(value)>
 *
 * Field lines are exactly four spaces of indent; the value is
 * `JSON.stringify`'d (so strings round-trip with quotes, objects /
 * arrays parse cleanly, numbers and `null` parse natively). This
 * shape is the contract the `inspect_gate_decisions` MCP tool relies
 * on to find `'gate decision'` records by `chatJid` / `messageId`.
 *
 * The parser is intentionally tolerant of:
 *   - blank lines between records (optional separator),
 *   - ANSI color codes that the file sink already strips (`stripAnsi`
 *     in `logger.ts`) but which historical rotated logs may still
 *     carry — we strip again as a belt-and-braces step,
 *   - records with no fields (header-only),
 *   - field values that fail `JSON.parse` (left as the raw string
 *     after the colon — the caller decides whether the record is
 *     usable).
 *
 * Today's date stamp on records is NOT in the line — the logger
 * emits time-of-day only (HH:mm:ss.SSS). For `since_minutes` /
 * absolute-time filtering, callers join with the file's mtime or
 * accept that the timestamps are wall-clock-of-the-day. The
 * `inspect_gate_decisions` use case is "answer about a recent
 * message", and rotation is **size-based** (`ORCHESTRATOR_LOG_MAX_
 * BYTES` in `src/logger.ts`, currently 10 MB), NOT age-based —
 * older records spill to `orchestrator.log.1` once the live file
 * crosses the threshold. The tool reads only the live file
 * (`orchestrator.log`); records rolled to `.1` aren't searchable
 * here.
 */

import fs from 'fs';

/** A single record parsed out of an orchestrator-log block. */
export interface ParsedLogRecord {
  /** `HH:mm:ss.SSS` from the header line, verbatim. */
  timestamp: string;
  /** Pino-style level uppercased (`INFO`, `WARN`, `ERROR`, `DEBUG`, `FATAL`). */
  level: string;
  /** PID stamped on the header line. */
  pid: number;
  /** Log message text from the header line. */
  msg: string;
  /**
   * Field name → parsed value. JSON-parseable values are decoded
   * (strings unquoted, objects/arrays inflated, numbers / `null`
   * native). Fields whose value isn't valid JSON are kept as the raw
   * post-colon string so the caller can still inspect them.
   */
  fields: Record<string, unknown>;
}

// eslint-disable-next-line no-control-regex -- intentional: strip ANSI SGR sequences from rotated log files
const ANSI_RE = /\x1b\[[0-9;]*m/g;
const HEADER_RE = /^\[(\d{2}:\d{2}:\d{2}\.\d{3})\] (\w+) \((\d+)\): (.*)$/;
const FIELD_RE = /^ {4}([^:]+): (.*)$/;

/**
 * Parse a string of orchestrator-log text into an array of records.
 *
 * Iterates line by line; a header line opens a new record, indented
 * field lines accumulate into it, and any other line (blank, broken,
 * or pre-header preamble) closes the current record so the next
 * header starts cleanly. Stripping ANSI is idempotent — already-
 * stripped lines pass through unchanged.
 */
export function parseHostLog(content: string): ParsedLogRecord[] {
  const out: ParsedLogRecord[] = [];
  let current: ParsedLogRecord | null = null;
  for (const rawLine of content.split('\n')) {
    const line = rawLine.replace(ANSI_RE, '');
    const headerMatch = HEADER_RE.exec(line);
    if (headerMatch) {
      if (current) out.push(current);
      current = {
        timestamp: headerMatch[1],
        level: headerMatch[2].toUpperCase(),
        pid: Number(headerMatch[3]),
        msg: headerMatch[4],
        fields: {},
      };
      continue;
    }
    if (!current) continue;
    const fieldMatch = FIELD_RE.exec(line);
    if (fieldMatch) {
      const key = fieldMatch[1];
      const rawValue = fieldMatch[2];
      let value: unknown = rawValue;
      try {
        value = JSON.parse(rawValue);
      } catch (err) {
        // Per `coding-policy: error-handling`: narrow to the expected
        // failure shape from `JSON.parse` — `SyntaxError` on malformed
        // JSON (a multi-line `err:` stack the logger renders specially
        // via `formatErr`, or a value that just doesn't happen to be
        // JSON). Keep the raw post-colon string so the caller can
        // still surface it; only the structured-fields contract
        // relies on JSON-parseability. Anything OTHER than
        // SyntaxError out of `JSON.parse` would indicate a runtime
        // bug in our own code (TypeError on a non-string arg, etc.)
        // and must propagate so the parser doesn't silently
        // misclassify it as "non-JSON value".
        if (!(err instanceof SyntaxError)) throw err;
      }
      current.fields[key] = value;
      continue;
    }
    // Blank or non-matching line — close the current record.
    out.push(current);
    current = null;
  }
  if (current) out.push(current);
  return out;
}

/**
 * Read and parse a host-log file from disk. Returns an empty array
 * if the path doesn't exist (callers treat "no log file" identically
 * to "log file with zero matching records" — both yield no results
 * for an inspect query).
 *
 * Does NOT tail or stream — reads the whole file at once. The live
 * `orchestrator.log` is rotation-capped via `ORCHESTRATOR_LOG_MAX_
 * BYTES` (see `logger.ts`), so a single read fits comfortably in
 * memory at any deployment we ship today.
 *
 * TOCTOU-safe against the logger's in-process rotation: the
 * `existsSync → readFileSync` window can race with the rotation
 * `renameSync` (logger.ts:188-210), so we also catch ENOENT from
 * the read itself. Both arms collapse to the same "no records"
 * outcome — a tool query landing the moment after a rotation just
 * sees an empty result instead of failing the whole IPC request.
 */
export function readHostLog(filePath: string): ParsedLogRecord[] {
  if (!fs.existsSync(filePath)) return [];
  let raw: string;
  try {
    raw = fs.readFileSync(filePath, 'utf-8');
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return [];
    throw err;
  }
  return parseHostLog(raw);
}

/** Filter input shape for `findGateDecisions`. */
export interface GateDecisionFilter {
  /** Required — chats not matching this JID are dropped. */
  chatJid: string;
  /** Optional — narrow to a single message id. */
  messageId?: string;
  /** Optional — at most this many records, most-recent first. Default 10. */
  limit?: number;
}

/**
 * Single normalized gate-decision record returned by
 * `findGateDecisions`. Mirrors the field set
 * `evaluateGateChain` writes via `logger.info(... 'gate decision')`.
 */
export interface GateDecisionRecord {
  timestamp: string;
  chatJid: string;
  messageId: string;
  groupFolder: string;
  finalDecision: 'allow' | 'deny';
  reason: string;
  chain: Array<{
    gate: string;
    decision: 'allow' | 'deny' | 'pass';
    reason: string;
  }>;
}

/**
 * Walk parsed records and return the `'gate decision'` ones matching
 * the filter, most-recent first. Ordering uses the file's natural
 * append-only sequence (last record in the parsed array is the most
 * recent write); we reverse-traverse to skip the bulk of the file
 * once `limit` is satisfied.
 *
 * Records whose shape doesn't match the contract (missing
 * `chatJid`, malformed `chain`) are skipped silently — a parser-
 * level mismatch is a code bug to surface elsewhere, not user-
 * facing noise on this read path.
 */
export function findGateDecisions(
  records: ParsedLogRecord[],
  filter: GateDecisionFilter,
): GateDecisionRecord[] {
  const limit = filter.limit ?? 10;
  const out: GateDecisionRecord[] = [];
  for (let i = records.length - 1; i >= 0 && out.length < limit; i--) {
    const r = records[i];
    if (r.msg !== 'gate decision') continue;
    const f = r.fields;
    if (typeof f.chatJid !== 'string' || f.chatJid !== filter.chatJid) continue;
    if (typeof f.messageId !== 'string') continue;
    if (filter.messageId !== undefined && f.messageId !== filter.messageId) {
      continue;
    }
    if (typeof f.groupFolder !== 'string') continue;
    if (f.finalDecision !== 'allow' && f.finalDecision !== 'deny') continue;
    if (typeof f.reason !== 'string') continue;
    if (!Array.isArray(f.chain)) continue;
    out.push({
      timestamp: r.timestamp,
      chatJid: f.chatJid,
      messageId: f.messageId,
      groupFolder: f.groupFolder,
      finalDecision: f.finalDecision,
      reason: f.reason,
      chain: f.chain as GateDecisionRecord['chain'],
    });
  }
  return out;
}
