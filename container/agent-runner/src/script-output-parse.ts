// Pure parser for precheck-script JSON output. Extracted from runScript
// so the wake-key contract has a regression test (#480 — silent gate-out
// caused by a key-spelling mismatch between runtime and skill emitters).
//
// Contract per `coding-policy: rules/script-delegation.md`:
//   `{"wake_agent": false, "data": {}}` — `wake_agent` is a boolean and
//   `data` is an object. The parser accepts an absent `data` (treats as
//   `{}`) but rejects any non-object value (string, number, null, array)
//   so a contract-violating skill surfaces as a parse failure rather
//   than passing through with a malformed payload.

export interface ScriptResult {
  wake_agent: boolean;
  data?: Record<string, unknown>;
}

export type ParseScriptOutcome =
  | { ok: true; result: ScriptResult }
  | {
      ok: false;
      reason:
        | 'empty'
        | 'invalid_json'
        | 'missing_wake_agent'
        | 'invalid_data_shape';
      lastLine?: string;
    };

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value) &&
    // Reject Date, Map, Set, etc. — JSON.parse only produces plain objects,
    // arrays, and primitives, so a non-Array object reaches this branch
    // exclusively as a plain dict.
    Object.getPrototypeOf(value) === Object.prototype
  );
}

export function parseScriptOutput(stdout: string): ParseScriptOutcome {
  const lines = stdout.trim().split('\n');
  const lastLine = lines[lines.length - 1];
  if (!lastLine) {
    return { ok: false, reason: 'empty' };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(lastLine);
  } catch {
    return { ok: false, reason: 'invalid_json', lastLine };
  }
  if (
    !parsed ||
    typeof parsed !== 'object' ||
    typeof (parsed as { wake_agent?: unknown }).wake_agent !== 'boolean'
  ) {
    return { ok: false, reason: 'missing_wake_agent', lastLine };
  }
  const data = (parsed as { data?: unknown }).data;
  if (data !== undefined && !isPlainObject(data)) {
    return { ok: false, reason: 'invalid_data_shape', lastLine };
  }
  return { ok: true, result: parsed as ScriptResult };
}
