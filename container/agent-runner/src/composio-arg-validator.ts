/**
 * #326 — Output schema validation on Composio tool args.
 *
 * Composio tool arguments are produced by the model and dispatched
 * as-is to Composio's REST endpoints. Without an outbound validator,
 * a model running under prompt-injection load can emit args that
 * exploit downstream parsers — header injection in `gmail.send`
 * (`subject = "Hi\n\nBcc: attacker@evil.com"`), embedded carriage
 * returns in Slack channels, oversize bodies that get truncated or
 * trip rate limits, and control characters that break log scrapers.
 *
 * #117 sanitizes incoming MCP tool RESULTS. This module is the
 * symmetric outbound check: a `PreToolUse` hook that validates per-
 * field constraints on the WAY OUT, before the bytes leave the
 * container.
 *
 * Scope (per #326 acceptance):
 *   - gmail send/reply: header fields (`subject`, `to`, `cc`, `bcc`,
 *     `recipient_email`) reject newlines + cap at RFC 5322 line limit;
 *     `body` capped at 100 KB and rejects control chars except \n \t.
 *   - slack post/send: header field (`channel`) rejects newlines;
 *     `text` capped at 100 KB and rejects control chars except \n \t.
 *
 * Tools outside this map pass through — the validator returns
 * `kind: 'allow'`. Github mutations, calendar create, and other
 * non-headered Composio writes don't have header-injection surfaces;
 * adding a row here is a one-line change when one materializes.
 *
 * The validator is tool-only — it does not look at provenance. Header
 * injection and oversize bodies are wrong regardless of who issued the
 * call. Operator-originated chains pay the same gate, but the gate is
 * cheap (a few regex tests + a length check) and the operator can
 * never legitimately produce a `\n` in a `subject`.
 */

/**
 * Field-level constraints. Either constraint can be set; both run if
 * both are set. `noNewlines` rejects `\r` and `\n` outright. `maxBytes`
 * is a UTF-8 byte cap (Buffer.byteLength), not a character count, so a
 * value with multi-byte characters can't sneak past a chars-based limit.
 * `noControlChars` rejects ASCII control chars (`< 0x20` and `0x7F`)
 * except the explicit allowlist — `\n` (0x0A) and `\t` (0x09).
 *
 * `mayBeArray` opts a field into accepting `string[]` — every element
 * is then validated independently under the same constraint. Default
 * (omitted / `false`) treats arrays as `wrong_type`. This flag exists
 * so body fields (`body`, `text`) — which Composio accepts only as
 * strings — can't be smuggled in as arrays to bypass the per-element
 * byte cap and ship N*100KB total payload.
 */
export interface FieldConstraint {
  noNewlines?: boolean;
  maxBytes?: number;
  noControlChars?: boolean;
  mayBeArray?: boolean;
}

/**
 * Mail header line limit per RFC 5322 §2.1.1: 998 octets between CRLF.
 * Used as the byte cap on header fields so a value the model produced
 * inline can't blow past the format limit even without a newline.
 */
const RFC_5322_HEADER_BYTES = 998;

/**
 * Body cap. Configurable here; Composio's own per-API limits are
 * higher (Gmail accepts ~25 MB attachments; Slack accepts ~40 KB
 * per `text` block) but a 100 KB cap is generous for typed content
 * while small enough to refuse pasted-in document blobs that almost
 * never come from operator intent.
 */
const BODY_MAX_BYTES = 100_000;

const HEADER_NO_NEWLINES: FieldConstraint = {
  noNewlines: true,
  maxBytes: RFC_5322_HEADER_BYTES,
};

/**
 * Recipient-list header — same single-line / RFC-5322-cap rules as
 * `HEADER_NO_NEWLINES`, but Composio accepts both `string` and
 * `string[]` shapes (`to: "a@x.io"` and `to: ["a@x.io", "b@y.io"]`
 * are both legal). `mayBeArray` opts each element into the same
 * per-string constraint; the size cap is per-element, which mirrors
 * RFC 5322's per-line limit (each address is its own header line).
 */
const HEADER_NO_NEWLINES_LIST: FieldConstraint = {
  noNewlines: true,
  maxBytes: RFC_5322_HEADER_BYTES,
  mayBeArray: true,
};

const BODY_LIMIT: FieldConstraint = {
  maxBytes: BODY_MAX_BYTES,
  noControlChars: true,
};

/**
 * Per-tool field rule table. Tool name is matched against `pattern`
 * (anchored regex). The first matching rule applies; later rules are
 * not inspected. Adding a new tool family is a one-row append.
 *
 * Field names mirror what Composio's gmail/slack actions accept:
 *   - gmail: `subject`, `recipient_email`, `to`, `cc`, `bcc`, `body`
 *   - slack: `channel`, `user`, `text`
 *
 * Fields that don't appear in a given call's input are simply
 * skipped — `gmail_reply_email` doesn't take `subject` (it's
 * inherited from the thread), so the rule's `subject` row never
 * fires when reply is the active tool.
 */
interface ToolRules {
  pattern: RegExp;
  fields: Record<string, FieldConstraint>;
}

const TOOL_RULES: ReadonlyArray<ToolRules> = [
  // Gmail send / reply / draft — every variant the egress-allowlist
  // already classifies as `gmail_send`. Recipient-list fields
  // (`to`, `cc`, `bcc`, `recipient_email`, `recipient`, `recipients`)
  // accept both `string` and `string[]` per Composio's input shape;
  // the array path validates every element under the same single-
  // line / RFC-5322-cap rule. `subject` and `body` are string-only —
  // arrays there would smuggle past the per-element body cap.
  {
    pattern: /^mcp__composio__gmail_(send|reply)\w*$/i,
    fields: {
      subject: HEADER_NO_NEWLINES,
      recipient_email: HEADER_NO_NEWLINES_LIST,
      recipient: HEADER_NO_NEWLINES_LIST,
      recipients: HEADER_NO_NEWLINES_LIST,
      to: HEADER_NO_NEWLINES_LIST,
      cc: HEADER_NO_NEWLINES_LIST,
      bcc: HEADER_NO_NEWLINES_LIST,
      body: BODY_LIMIT,
    },
  },
  // Slack post / send-DM — channel is the header surface, text is
  // the body. `user` (DM target) gets the same header treatment as
  // channel because newlines / oversize there confuse Slack's parser.
  // None of the Slack fields legitimately come as arrays.
  {
    pattern: /^mcp__composio__slack_(post|send)\w*$/i,
    fields: {
      channel: HEADER_NO_NEWLINES,
      user: HEADER_NO_NEWLINES,
      text: BODY_LIMIT,
    },
  },
];

export type Violation =
  | 'newlines_in_header'
  | 'header_too_long'
  | 'body_too_long'
  | 'control_chars_in_body'
  | 'wrong_type';

export type ValidationDecision =
  | { kind: 'allow' }
  | {
      kind: 'deny';
      field: string;
      violation: Violation;
      reason: string;
    };

/**
 * Validate the args for a Composio tool call. Returns the first
 * violation found, or `kind: 'allow'` if the tool is unmapped or
 * every constrained field passes.
 *
 * The "first violation wins" pattern is intentional — the model only
 * needs to know about ONE problem to fix and retry; surfacing all of
 * them at once produces verbose deny messages the model has to parse
 * back out. A second call exposes a second violation if one exists.
 */
export function validateComposioArgs(
  toolName: string,
  toolInput: unknown,
): ValidationDecision {
  if (typeof toolName !== 'string' || toolName.length === 0) {
    return { kind: 'allow' };
  }
  if (!toolInput || typeof toolInput !== 'object') {
    return { kind: 'allow' };
  }

  const rule = TOOL_RULES.find((r) => r.pattern.test(toolName));
  if (!rule) return { kind: 'allow' };

  const input = toolInput as Record<string, unknown>;

  for (const [fieldName, constraint] of Object.entries(rule.fields)) {
    const value = input[fieldName];
    if (value === undefined || value === null) continue;

    // Composio recipient-list header fields (`to`, `cc`, `bcc`,
    // `recipient_email`, `recipient`, `recipients`) accept both
    // `string` and `string[]` shapes. Body fields (`body`, `text`)
    // and non-list headers (`subject`, `channel`, `user`) are
    // string-only — accepting arrays there would let an injection
    // smuggle past the per-element byte cap by shipping multiple
    // 100KB chunks under one field name.
    if (Array.isArray(value)) {
      if (constraint.mayBeArray !== true) {
        return denyForField(
          toolName,
          fieldName,
          'wrong_type',
          `value is array, expected string (this field does not accept arrays)`,
        );
      }
      for (let i = 0; i < value.length; i++) {
        const elem = value[i];
        if (typeof elem !== 'string') {
          return denyForField(
            toolName,
            fieldName,
            'wrong_type',
            `array element at index ${i} is ${typeof elem}, expected string`,
          );
        }
        const v = checkString(elem, constraint);
        if (v) {
          return denyForField(
            toolName,
            `${fieldName}[${i}]`,
            v.violation,
            v.detail,
          );
        }
      }
      continue;
    }

    if (typeof value !== 'string') {
      return denyForField(
        toolName,
        fieldName,
        'wrong_type',
        `value is ${typeof value}, expected ${constraint.mayBeArray ? 'string or string[]' : 'string'}`,
      );
    }

    const v = checkString(value, constraint);
    if (v) return denyForField(toolName, fieldName, v.violation, v.detail);
  }

  return { kind: 'allow' };
}

interface StringCheckResult {
  violation: Violation;
  detail: string;
}

function checkString(
  value: string,
  c: FieldConstraint,
): StringCheckResult | null {
  if (c.noNewlines && /[\r\n]/.test(value)) {
    return {
      violation: 'newlines_in_header',
      detail: 'value contains \\r or \\n; header fields must be single-line',
    };
  }
  if (c.maxBytes !== undefined) {
    const byteLen = Buffer.byteLength(value, 'utf8');
    if (byteLen > c.maxBytes) {
      const isHeader = c.noNewlines === true;
      return {
        violation: isHeader ? 'header_too_long' : 'body_too_long',
        detail:
          `value is ${byteLen} bytes; cap is ${c.maxBytes} bytes ` +
          `(measured as UTF-8)`,
      };
    }
  }
  if (c.noControlChars) {
    // ASCII control chars are 0x00-0x1F and 0x7F. Allow only 0x09 (\t)
    // and 0x0A (\n); a body field may legitimately contain those. \r
    // (0x0D) is included in the reject set because CRLF in a body is
    // the common header-spoof pattern even when the field isn't a header.
    const m = value.match(/[\x00-\x08\x0B-\x1F\x7F]/);
    if (m) {
      const code = m[0].charCodeAt(0);
      return {
        violation: 'control_chars_in_body',
        detail:
          `value contains control char 0x${code.toString(16).padStart(2, '0').toUpperCase()} ` +
          `at offset ${m.index}; only \\t (0x09) and \\n (0x0A) are allowed`,
      };
    }
  }
  return null;
}

function denyForField(
  toolName: string,
  fieldName: string,
  violation: Violation,
  detail: string,
): ValidationDecision {
  return {
    kind: 'deny',
    field: fieldName,
    violation,
    reason:
      `composio_arg_validator: ${toolName} field '${fieldName}' rejected ` +
      `(${violation}). ${detail}. Re-issue the call with a valid value; ` +
      `if the value came from external content, the safe fix is to ` +
      `summarize / quote rather than pass through.`,
  };
}
