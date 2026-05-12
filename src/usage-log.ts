/**
 * Detailed Anthropic API usage logger.
 *
 * Captures per-request `usage` from `/v1/messages` responses at the
 * credential proxy and appends one JSON line per call to
 * `logs/usage.jsonl`. This is the authoritative source of truth for
 * cost accounting — same numbers Anthropic bills against.
 *
 * The proxy is on the critical path for every API call, so this module
 * MUST NEVER throw or block the response stream. All write errors are
 * swallowed with a warn log.
 */
import { promises as fsp } from 'fs';
import { dirname, join } from 'path';

import { logger } from './logger.js';

/** Pricing in $ per 1M tokens. 5m and 1h cache writes are priced separately. */
export interface ModelPricing {
  in_: number;
  out: number;
  cache_r: number;
  cache_c_5m: number;
  cache_c_1h: number;
}

/**
 * Anthropic public pricing (as of 2026-05). Update when Anthropic
 * changes prices.
 *
 * Source: https://www.anthropic.com/pricing
 *
 * Unknown models go through `resolvePricing()` which tries (in order):
 *   1. Exact match in this table.
 *   2. Strip a `-YYYYMMDD` date suffix and retry.
 *   3. Family-prefix fallback: pick the latest-known entry in the same
 *      family (opus / sonnet / haiku). #479 sub-#2: catches the "next
 *      major bump" case (e.g. `claude-opus-4-8`) where Sonnet's prior
 *      blanket fallback would have logged ~5× too low for Opus traffic.
 *   4. Last-resort fallback to Sonnet with a warn log, AND the resulting
 *      record is flagged `cost_unknown: true` so cost reports surface
 *      the gap instead of silently underreporting.
 */
export const PRICING: Record<string, ModelPricing> = {
  'claude-sonnet-4-6': {
    in_: 3,
    out: 15,
    cache_r: 0.3,
    cache_c_5m: 3.75,
    cache_c_1h: 6,
  },
  'claude-sonnet-4-5': {
    in_: 3,
    out: 15,
    cache_r: 0.3,
    cache_c_5m: 3.75,
    cache_c_1h: 6,
  },
  'claude-haiku-4-5': {
    in_: 1,
    out: 5,
    cache_r: 0.1,
    cache_c_5m: 1.25,
    cache_c_1h: 2,
  },
  'claude-haiku-4-5-20251001': {
    in_: 1,
    out: 5,
    cache_r: 0.1,
    cache_c_5m: 1.25,
    cache_c_1h: 2,
  },
  'claude-opus-4-7': {
    in_: 5,
    out: 25,
    cache_r: 0.5,
    cache_c_5m: 6.25,
    cache_c_1h: 10,
  },
};

const FALLBACK_MODEL = 'claude-sonnet-4-6';

/**
 * Result of a pricing lookup, carrying a flag so the caller can mark
 * records that resorted to the cross-family last-resort fallback.
 * `approximate` is true for family-prefix matches (cost is close but
 * not exact); `cost_unknown` is true only when no family matched and
 * we fell back to Sonnet — those rows should be excluded from precise
 * cost totals.
 */
export interface PricingResult {
  pricing: ModelPricing;
  /** True for any non-exact match (family fallback or last-resort). */
  approximate: boolean;
  /** True only when no family matched — Sonnet last-resort. */
  cost_unknown: boolean;
}

const KNOWN_FAMILIES = ['opus', 'sonnet', 'haiku'] as const;
type Family = (typeof KNOWN_FAMILIES)[number];

/**
 * Detect the model family from a name like `claude-opus-4-8` or
 * `claude-sonnet-4-6-20260512`. Returns null if no known family
 * substring is found.
 */
function detectFamily(model: string): Family | null {
  for (const fam of KNOWN_FAMILIES) {
    if (model.includes(`-${fam}-`) || model.includes(`-${fam}`)) return fam;
  }
  return null;
}

/**
 * Among PRICING entries belonging to a given family, pick the one with
 * the highest version. Compares **numeric version components** instead
 * of lexicographic strings so `claude-x-4-10` ranks above `claude-x-4-9`
 * once two-digit minor versions show up. Returns null if the family has
 * no entries (PRICING is misconfigured).
 */
function latestEntryInFamily(family: Family): string | null {
  const entries = Object.keys(PRICING).filter(
    (k) => k.includes(`-${family}-`) || k.endsWith(`-${family}`),
  );
  if (entries.length === 0) return null;
  // Drop date suffix and split on `-` to extract the numeric version
  // tail (everything after the family token). `claude-sonnet-4-6` →
  // [4, 6]; `claude-haiku-4-5-20251001` → [4, 5]; non-numeric tail
  // segments sort below numeric ones (treated as -Infinity).
  const versionTuple = (model: string): number[] => {
    const stripped = model.replace(/-\d{8}$/, '');
    const idx = stripped.indexOf(`-${family}-`);
    if (idx === -1) return [];
    const tail = stripped.slice(idx + family.length + 2);
    return tail.split('-').map((segment) => {
      const n = Number(segment);
      return Number.isFinite(n) ? n : -Infinity;
    });
  };
  // Element-wise descending compare — `[4,10]` ranks above `[4,9]`.
  entries.sort((a, b) => {
    const ta = versionTuple(a);
    const tb = versionTuple(b);
    const len = Math.max(ta.length, tb.length);
    for (let i = 0; i < len; i++) {
      const av = ta[i] ?? -Infinity;
      const bv = tb[i] ?? -Infinity;
      if (av !== bv) return bv - av;
    }
    return 0;
  });
  return entries[0];
}

/**
 * Resolve a model name to a pricing entry plus accuracy flags. Never
 * throws.
 *
 * Sub-#2 of #479: replaced the old "fall back to Sonnet for any
 * unknown model" with family-prefix fallback. The blanket Sonnet
 * fallback would log ~5× too low for any future Opus traffic
 * (`claude-opus-4-8` etc.) until the pricing table caught up.
 */
export function resolvePricing(model: string): PricingResult {
  if (PRICING[model]) {
    return { pricing: PRICING[model], approximate: false, cost_unknown: false };
  }
  // Strip a trailing -YYYYMMDD date suffix and retry exact.
  const stripped = model.replace(/-\d{8}$/, '');
  if (PRICING[stripped]) {
    return {
      pricing: PRICING[stripped],
      approximate: false,
      cost_unknown: false,
    };
  }
  // Family fallback — pick the latest known entry in the same family
  // so a next-version bump (`claude-opus-4-8`) is priced as Opus, not
  // Sonnet, until the table is updated.
  const family = detectFamily(model);
  if (family) {
    const latest = latestEntryInFamily(family);
    if (latest) {
      logger.warn(
        { model, family, fallbackEntry: latest },
        'usage-log: unknown model, family-prefix fallback to latest known entry in family',
      );
      return {
        pricing: PRICING[latest],
        approximate: true,
        cost_unknown: false,
      };
    }
  }
  // No family detected — last-resort Sonnet fallback. Flag the record
  // so cost reports can surface the gap.
  logger.warn(
    { model },
    'usage-log: unknown model with no detectable family — using Sonnet last-resort fallback; record flagged cost_unknown',
  );
  return {
    pricing: PRICING[FALLBACK_MODEL],
    approximate: true,
    cost_unknown: true,
  };
}

/** Token counts captured from an Anthropic response `usage` field. */
export interface TokenCounts {
  in_: number;
  out: number;
  cache_r: number;
  cache_c_5m: number;
  cache_c_1h: number;
}

/**
 * Compute cost in microcents (integer; multiply by 1e-8 to get dollars,
 * i.e. divide by 1e8). The field is named `cost_micro` historically but
 * the unit is microcents, NOT micro-USD — see derivation below. Consumers
 * that read this value MUST divide by 1e8, not 1e6.
 *
 * Derivation:
 *   $ = sum(tokens_i * price_i_per_MTok / 1e6)
 *   cents = $ * 100
 *   microcents = cents * 1e6 = sum(tokens_i * price_i) * 100
 *
 * The `* 100` factor IS the entire conversion: one microcent equals
 * 1e-8 dollars, prices are per million tokens ($/1e6), and tokens *
 * price/1e6 = $, so tokens * price * 100 = microcents.
 */
export function computeCostMicro(
  tokens: TokenCounts,
  pricing: ModelPricing,
): number {
  const dollars =
    (tokens.in_ * pricing.in_ +
      tokens.out * pricing.out +
      tokens.cache_r * pricing.cache_r +
      tokens.cache_c_5m * pricing.cache_c_5m +
      tokens.cache_c_1h * pricing.cache_c_1h) /
    1e6;
  return Math.round(dollars * 1e8);
}

/** Container attribution context, looked up from per-spawn token. */
export interface ContainerContext {
  group: string;
  tier: 'main' | 'trusted' | 'untrusted' | 'classifier';
  session: string;
  task_id: string | null;
  /**
   * #479 sub-#1: trigger message_id when an inbound user message
   * caused the spawn, null otherwise (scheduled tasks, IPC scripts,
   * housekeeping). The orchestrator wires this from
   * `evaluateGateChain`'s `allowedMessageId` — the message that
   * actually cleared the gate verdict and triggered the spawn — and
   * falls back to `replyToMessageId` (the reply target) when no gate
   * chain ran. Per-message attribution lets cost reports break down
   * spend by specific Telegram message instead of stopping at
   * session-level.
   */
  message_id: string | null;
}

/** Single line written to logs/usage.jsonl. */
export interface UsageRecord {
  ts: string;
  group: string;
  tier: string;
  session: string;
  task_id: string | null;
  /**
   * #479 sub-#1: trigger message_id (channel-side, e.g. Telegram
   * message id) for user-initiated spawns, null for scheduled tasks /
   * IPC scripts / housekeeping. Lets cost reports break down spend by
   * specific message instead of stopping at session granularity.
   */
  message_id: string | null;
  model: string;
  api_id: string | null;
  in: number;
  out: number;
  cache_r: number;
  cache_c_5m: number;
  cache_c_1h: number;
  /** Cost in microcents (1e-8 USD per unit). Divide by 1e8 for dollars. */
  cost_micro: number;
  dur_ms: number;
  /**
   * #479 sub-#2: present (`true`) only when the pricing fallback
   * resorted to the cross-family last-resort entry — i.e. the model
   * has no detectable family and `cost_micro` is computed at Sonnet
   * rates. Cost aggregators MUST exclude or call out these rows so an
   * unmodelled family doesn't silently underreport totals. Omitted
   * (undefined) for exact-match and same-family fallback rows.
   */
  cost_unknown?: boolean;
  /**
   * #479 sub-#2: present (`true`) when pricing came from a same-family
   * fallback (e.g. `claude-opus-4-8` → `claude-opus-4-7` rates).
   * `cost_micro` is in the right ballpark but not exact. Aggregators
   * may include these rows in totals while flagging them as
   * approximate.
   */
  cost_approximate?: boolean;
}

/**
 * Build a usage record from a response body and context. Returns null
 * if the body has no usable `usage` field.
 *
 * Handles both shapes:
 *   - non-streaming: top-level JSON with `usage` and `id`
 *   - streaming SSE: a series of `data: {...}` events; the cumulative
 *     `usage` is on the `message_delta` event (the start event has the
 *     input tokens; the delta has the final output count). We merge
 *     fields across events.
 */
export function parseUsageFromBody(
  body: string,
  ctx: ContainerContext,
  durMs: number,
  fallbackModel: string | null,
): UsageRecord | null {
  // Try non-streaming first (cheap; no scan needed).
  if (body.length > 0 && body.charCodeAt(0) === 0x7b /* '{' */) {
    try {
      const parsed = JSON.parse(body);
      if (parsed && parsed.usage) {
        return buildUsageRecord(
          parsed.usage,
          parsed.model || fallbackModel || 'unknown',
          parsed.id || null,
          ctx,
          durMs,
        );
      }
    } catch {
      // Fall through to SSE parse.
    }
  }

  // SSE: scan for the message_start (carries id, model, partial usage)
  // and message_delta (carries final cumulative usage). The event lines
  // look like `event: message_delta\ndata: {...}\n\n`. We parse all
  // data: payloads and merge usage fields, preferring the latest.
  let mergedUsage: Record<string, number> | null = null;
  let model: string | null = null;
  let apiId: string | null = null;

  // Each event is separated by a blank line. Real-world SSE traffic
  // mixes LF and CRLF (the spec allows both, and Anthropic's edge
  // sometimes returns CRLF). Split on `\r?\n\r?\n` so either line
  // ending works; same for the per-event line split. This was a
  // suspected silent-failure root cause when 37 production captures
  // came back empty after the post-#487 deploy.
  const events = body.split(/\r?\n\r?\n/);
  for (const evt of events) {
    const dataLine = evt.split(/\r?\n/).find((l) => l.startsWith('data: '));
    if (!dataLine) continue;
    const json = dataLine.slice(6);
    let parsed: unknown;
    try {
      parsed = JSON.parse(json);
    } catch {
      continue;
    }
    if (!parsed || typeof parsed !== 'object') continue;
    const obj = parsed as {
      type?: string;
      message?: { id?: string; model?: string; usage?: Record<string, number> };
      usage?: Record<string, number>;
    };
    if (obj.type === 'message_start' && obj.message) {
      apiId = obj.message.id ?? apiId;
      model = obj.message.model ?? model;
      if (obj.message.usage)
        mergedUsage = { ...(mergedUsage || {}), ...obj.message.usage };
    } else if (obj.type === 'message_delta' && obj.usage) {
      mergedUsage = { ...(mergedUsage || {}), ...obj.usage };
    }
  }

  if (!mergedUsage) return null;
  return buildUsageRecord(
    mergedUsage,
    model || fallbackModel || 'unknown',
    apiId,
    ctx,
    durMs,
  );
}

/**
 * Anthropic's `usage` shape uses these keys:
 *   input_tokens, output_tokens, cache_read_input_tokens,
 *   cache_creation_input_tokens, and the breakdown
 *   cache_creation: { ephemeral_5m_input_tokens, ephemeral_1h_input_tokens }
 *
 * When `cache_creation` is absent we credit the total to 5m (which is
 * the SDK's default TTL).
 *
 * Exported so non-proxy call sites (e.g. the orchestrator-side Haiku
 * classifier in `gates/haiku-classifier.ts`) can append the same shape
 * without going through the credential proxy.
 */
export function buildUsageRecord(
  usage: Record<string, number | Record<string, number>>,
  model: string,
  apiId: string | null,
  ctx: ContainerContext,
  durMs: number,
): UsageRecord {
  const numField = (k: string): number => {
    const v = usage[k];
    return typeof v === 'number' ? v : 0;
  };
  const cacheCreation = usage.cache_creation;
  let cache_c_5m = 0;
  let cache_c_1h = 0;
  if (cacheCreation && typeof cacheCreation === 'object') {
    const cc = cacheCreation as Record<string, number>;
    cache_c_5m =
      typeof cc.ephemeral_5m_input_tokens === 'number'
        ? cc.ephemeral_5m_input_tokens
        : 0;
    cache_c_1h =
      typeof cc.ephemeral_1h_input_tokens === 'number'
        ? cc.ephemeral_1h_input_tokens
        : 0;
  } else {
    // No breakdown — credit everything to 5m (the SDK default TTL).
    cache_c_5m = numField('cache_creation_input_tokens');
  }

  const tokens: TokenCounts = {
    in_: numField('input_tokens'),
    out: numField('output_tokens'),
    cache_r: numField('cache_read_input_tokens'),
    cache_c_5m,
    cache_c_1h,
  };
  const priced = resolvePricing(model);
  const cost_micro = computeCostMicro(tokens, priced.pricing);

  const record: UsageRecord = {
    ts: new Date().toISOString(),
    group: ctx.group,
    tier: ctx.tier,
    session: ctx.session,
    task_id: ctx.task_id,
    message_id: ctx.message_id,
    model,
    api_id: apiId,
    in: tokens.in_,
    out: tokens.out,
    cache_r: tokens.cache_r,
    cache_c_5m: tokens.cache_c_5m,
    cache_c_1h: tokens.cache_c_1h,
    cost_micro,
    dur_ms: durMs,
  };
  if (priced.cost_unknown) record.cost_unknown = true;
  if (priced.approximate) record.cost_approximate = true;
  return record;
}

/**
 * Resolve the JSONL log path. Defaults to `logs/usage.jsonl` under the
 * orchestrator's cwd; override with `USAGE_LOG_PATH` for tests or to
 * relocate. Both the credential proxy and the host-side Haiku classifier
 * use this so they agree on a single sink.
 */
export function resolveUsageLogPath(): string {
  return process.env.USAGE_LOG_PATH || join('logs', 'usage.jsonl');
}

/**
 * Append a usage record as a JSON line. Fail-safe: any IO error is
 * caught and logged at warn level so the proxy response path is never
 * affected. The directory is created on first call.
 */
let dirEnsured = false;
export async function appendUsageRecord(
  path: string,
  record: UsageRecord,
): Promise<void> {
  try {
    if (!dirEnsured) {
      await fsp.mkdir(dirname(path), { recursive: true });
      dirEnsured = true;
    }
    await fsp.appendFile(path, JSON.stringify(record) + '\n', 'utf8');
  } catch (err) {
    logger.warn({ err, path }, 'usage-log: append failed');
  }
}

/**
 * Silent-zero-output guard (#479 sub-#3).
 *
 * #126 incident: 13 minutes of post-restart `/v1/messages` traffic
 * with zero JSONL lines, no warnings, just an empty file. Worst kind
 * of failure for a metrics pipeline.
 *
 * The proxy hooks two events:
 *   - `noteMessagesRequest()` on every captured /v1/messages POST.
 *   - `noteCaptureWrite()` when a record is successfully built and
 *     queued for append.
 *
 * `checkSilentZero(now)` is called from the orchestrator's existing
 * periodic loop; it returns a diagnostic string (or null when healthy)
 * and resets its alert clock so a single broken state isn't logged
 * every tick.
 */
let messagesSeen = 0;
let capturesWritten = 0;
let firstMessageMs: number | null = null;
let lastAlertedAtMessages = -1;

const SILENT_ZERO_GRACE_MS = 60 * 1000; // 1 minute warmup after first request

export function noteMessagesRequest(): void {
  messagesSeen += 1;
  if (firstMessageMs === null) firstMessageMs = Date.now();
}

export function noteCaptureWrite(): void {
  capturesWritten += 1;
}

export interface SilentZeroDiagnostic {
  messagesSeen: number;
  capturesWritten: number;
  ageMs: number;
}

/**
 * Return a diagnostic if the proxy has handled at least one
 * /v1/messages POST but has produced no JSONL records for longer than
 * `SILENT_ZERO_GRACE_MS`. Otherwise return null. The check is
 * edge-triggered against `messagesSeen`: once an alert fires for a
 * given count, it won't fire again until new messages arrive — so a
 * 30-min orchestrator loop won't generate 30 duplicate WARN lines for
 * the same broken state.
 */
export function checkSilentZero(
  now: number = Date.now(),
): SilentZeroDiagnostic | null {
  if (firstMessageMs === null) return null;
  if (capturesWritten > 0) return null;
  const age = now - firstMessageMs;
  if (age < SILENT_ZERO_GRACE_MS) return null;
  if (messagesSeen === lastAlertedAtMessages) return null;
  lastAlertedAtMessages = messagesSeen;
  return {
    messagesSeen,
    capturesWritten,
    ageMs: age,
  };
}

/** Reset the dir-ensured cache + silent-zero counters. Test-only. */
export function _resetUsageLogState(): void {
  dirEnsured = false;
  messagesSeen = 0;
  capturesWritten = 0;
  firstMessageMs = null;
  lastAlertedAtMessages = -1;
}
