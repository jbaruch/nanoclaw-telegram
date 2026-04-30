/**
 * Typed source taxonomy for the `<untrusted-input source="...">` wrap (#321,
 * consumed by #322's walk-back).
 *
 * The wrap marks external/untrusted-provenance content entering the agent's
 * context. The `source=` attribute uses a typed `prefix:value` pair so the
 * #322 capability ACL can apply different rules per source kind without
 * parsing free-form strings.
 *
 * Every emitter — orchestrator-side prompt wrap, MCP PostToolUse wrap,
 * agent-browser script, built-in-tool sentinel — calls these helpers so
 * the prefix taxonomy stays consistent across the four #321 PRs.
 */

export type SourcePrefix =
  | 'untrusted-container'
  | 'cross-group'
  | 'web'
  | 'gmail'
  | 'calendar'
  | 'slack'
  | 'github'
  | 'tessl'
  | 'file'
  | 'agent-browser';

export function formatSource(prefix: SourcePrefix, value: string): string {
  return `${prefix}:${value}`;
}

export function wrapUntrustedInput(
  content: string,
  prefix: SourcePrefix,
  value: string,
): string {
  const sourceAttr = escapeAttr(formatSource(prefix, value));
  return `<untrusted-input source="${sourceAttr}">\n${content}\n</untrusted-input>`;
}

function escapeAttr(s: string): string {
  return s.replace(/"/g, '&quot;').replace(/[\r\n]+/g, ' ');
}
