#!/usr/bin/env -S node --experimental-strip-types
/**
 * aye-confirm — host-side CLI for #324.
 *
 * Mints a single-use, scoped, TTL'd confirmation token and appends it
 * to /workspace/trusted/aye_confirm_tokens.json (or the path supplied
 * via $AYE_CONFIRM_TOKEN_PATH). The agent's PreToolUse hook in
 * `container/agent-runner/src/index.ts` consumes the token when an
 * untrusted-provenance chain attempts a matching destructive op.
 *
 * Usage:
 *
 *   aye-confirm --scope <scope> --reason '<text>' --ttl <duration>
 *
 *   --scope    one of: nuke_session, nuke_chat, github_backup,
 *              set_trusted, set_trigger, set_agent_model,
 *              set_maintenance_agent_model, set_task_agent_model,
 *              register_group, unregister_group, promote_staging,
 *              push_staged_to_branch, tessl_update,
 *              schedule_task_harness, egress_allowlist_mutation,
 *              file_delete_outside_group
 *   --reason   short audit-trail prose
 *   --ttl      duration like 5m, 1h, 30s, 7d. Default: 5m
 *
 *   --chat-jid (optional) — restricts the token to one chat. v1 stores
 *              the value but does NOT enforce it (the agent hook reads
 *              `chat_jid` for telemetry only).
 *
 * The minted token is printed to stdout; everything else goes to stderr.
 */

import * as crypto from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { pathToFileURL } from 'url';
import {
  ConfirmationToken,
  DestructiveScope,
  loadConfirmationTokens,
  parseDuration,
  saveConfirmationTokens,
} from '../container/agent-runner/src/confirmation-tokens.ts';

const VALID_SCOPES: ReadonlyArray<DestructiveScope> = [
  'nuke_session',
  'nuke_chat',
  'github_backup',
  'set_trusted',
  'set_trigger',
  'set_agent_model',
  'set_maintenance_agent_model',
  'set_task_agent_model',
  'register_group',
  'unregister_group',
  'promote_staging',
  'push_staged_to_branch',
  'tessl_update',
  'schedule_task_harness',
  'egress_allowlist_mutation',
  'file_delete_outside_group',
];

interface ParsedArgs {
  scope: DestructiveScope;
  reason: string;
  ttl: string;
  chatJid?: string;
}

function usage(exitCode: number): never {
  const stream = exitCode === 0 ? process.stdout : process.stderr;
  stream.write(
    `aye-confirm — mint a destructive-op confirmation token\n\n` +
    `Usage:\n` +
    `  aye-confirm --scope <scope> --reason '<text>' [--ttl 5m] [--chat-jid <jid>]\n\n` +
    `Valid scopes:\n` +
    VALID_SCOPES.map((s) => `  - ${s}`).join('\n') +
    `\n\nThe minted token prints to stdout. Paste it into the chat (or ` +
    `let the agent retry the gated tool — it'll consume the token ` +
    `automatically on the next attempt).\n`,
  );
  process.exit(exitCode);
}

export function parseArgs(argv: ReadonlyArray<string>): ParsedArgs {
  let scope: string | undefined;
  let reason: string | undefined;
  let ttl = '5m';
  let chatJid: string | undefined;
  const consumeValue = (flag: string, i: number): string => {
    const v = argv[i + 1];
    if (v === undefined || v.startsWith('--')) {
      process.stderr.write(`error: ${flag} requires a value\n`);
      usage(2);
    }
    return v as string;
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '-h' || a === '--help') usage(0);
    else if (a === '--scope') {
      scope = consumeValue('--scope', i);
      i++;
    } else if (a === '--reason') {
      reason = consumeValue('--reason', i);
      i++;
    } else if (a === '--ttl') {
      ttl = consumeValue('--ttl', i);
      i++;
    } else if (a === '--chat-jid') {
      chatJid = consumeValue('--chat-jid', i);
      i++;
    } else {
      process.stderr.write(`unknown argument: ${a}\n`);
      usage(2);
    }
  }
  if (!scope) {
    process.stderr.write('error: --scope is required\n');
    usage(2);
  }
  if (!reason) {
    process.stderr.write('error: --reason is required\n');
    usage(2);
  }
  if (!VALID_SCOPES.includes(scope as DestructiveScope)) {
    process.stderr.write(`error: invalid --scope '${scope}'\n`);
    usage(2);
  }
  // Validate ttl by parsing; throws with a clear message on bad input.
  parseDuration(ttl);
  return { scope: scope as DestructiveScope, reason: reason as string, ttl, chatJid };
}

/**
 * 16 bytes = 32 hex chars. Plenty for a single-use, short-TTL value
 * that's not a long-lived secret. Operator pastes it; brevity wins.
 *
 * `randomBytesFn` is injected so tests can assert the deterministic
 * transform (n=16 → hex of those bytes) without spying on
 * `crypto.randomBytes` — ESM forbids spying on namespace exports
 * (see vitest's "Cannot redefine property" error). Production
 * callers pass nothing and get the real `crypto.randomBytes`.
 */
export function generateToken(
  randomBytesFn: (n: number) => Buffer = crypto.randomBytes,
): string {
  return randomBytesFn(16).toString('hex');
}

export function mintToken(args: ParsedArgs): ConfirmationToken {
  const now = Date.now();
  const ttlMs = parseDuration(args.ttl);
  return {
    token: generateToken(),
    scope: args.scope,
    reason: args.reason,
    issued_at: new Date(now).toISOString(),
    expires_at: new Date(now + ttlMs).toISOString(),
    used: false,
    ...(args.chatJid ? { chat_jid: args.chatJid } : {}),
  };
}

function resolveTokenPath(): string {
  const env = process.env.AYE_CONFIRM_TOKEN_PATH;
  if (env && env.length > 0) return env;
  return '/workspace/trusted/aye_confirm_tokens.json';
}

function ensureParentDir(p: string): void {
  const dir = path.dirname(p);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const tokenPath = resolveTokenPath();
  ensureParentDir(tokenPath);

  const existing = loadConfirmationTokens(fs, tokenPath);
  const fresh = mintToken(args);
  const next = [...existing, fresh];
  saveConfirmationTokens(fs, tokenPath, next);

  // Token to stdout (single line, easy to copy); telemetry to stderr.
  // Operator-supplied `reason` is intentionally OMITTED from the
  // telemetry line — `reason` is free text that may include sensitive
  // material (chat content, names, paths) and is forbidden by
  // `jbaruch/coding-policy: no-secrets`. The reason is still written
  // into the token file (the operator put it there, the operator can
  // read it back); telemetry stays on bounded non-sensitive fields.
  process.stdout.write(fresh.token + '\n');
  process.stderr.write(
    `aye-confirm: minted token for scope=${fresh.scope} ttl=${args.ttl} ` +
    `expires_at=${fresh.expires_at}\n`,
  );
}

if (
  // ESM entry-point guard. Use `pathToFileURL` because `process.argv[1]`
  // can be a relative path; comparing `file://${process.argv[1]}`
  // directly fails for relative paths and would silently turn the CLI
  // into a no-op. Skipped when imported by tests.
  process.argv[1] &&
  import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href
) {
  main();
}
