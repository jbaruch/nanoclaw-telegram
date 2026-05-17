/**
 * #576 — Enriched `<context>` tag for agent anchoring.
 *
 * Pure builder + formatter for the per-invocation context block the
 * orchestrator prepends to every prompt. Replaces the pre-#576
 * single-attribute `<context timezone="..." />` with a richer tag
 * that names the user's local frame so the agent can ground
 * relative phrasings (today / yesterday / now / here) in actual
 * coordinates instead of the server clock.
 *
 * Inputs are injected (DB rows + clock) so this module stays pure
 * and testable. The orchestrator's `index.ts` assembles the inputs
 * once per agent invocation and calls these two functions.
 *
 * The mapping from `tz-resolver.ts`'s precise `TzResolverSource`
 * union to the agent-facing four-value source label
 * (`location` / `segment` / `home_fallback` / `container_default`)
 * collapses a few resolver internals that the agent doesn't need
 * to differentiate: walker-coord-lookup-failed and walker-stale
 * both surface as `segment` (the walker drove the answer using a
 * tz_state segment row) or `home_fallback` (the walker had no
 * covering segment and returned `home_tz`). The walker doesn't
 * expose its branch directly, so the segment-vs-home distinction
 * is reconstructed by comparing the resolved tz to `home_tz` — a
 * corner case (user genuinely IN home tz with a covering segment)
 * mislabels as `home_fallback`, but the operational outcome is
 * identical and the rule (`local-context-anchoring`) doesn't
 * branch on that distinction anyway.
 */

import { LocationRecord } from './types.js';
import {
  TripitSegment,
  walkTzSegments,
  getLatestLocationForSender,
  readTzStateForContext,
} from './db.js';
import { resolveCurrentTz, TzResolverSource } from './tz-resolver.js';
import { resolveTimezone } from './timezone.js';
import { escapeXml } from './router.js';

export type AgentContextTimezoneSource =
  // Fresh shared-location pin (<4h old) drove the resolution.
  | 'location'
  // Walker fallback used a covering segment from `tz_state.segments`.
  | 'segment'
  // Walker fallback returned `home_tz` (no covering segment).
  | 'home_fallback'
  // Neither a location row nor segment data exists — the orchestrator
  // didn't run the resolver and emitted the container's configured
  // `TZ` instead. Agent should treat answers framed in this tz as
  // best-effort.
  | 'container_default';

export interface AgentContext {
  utc_datetime: string;
  local_datetime: string;
  local_date: string;
  weekday: string;
  timezone: string;
  timezone_source: AgentContextTimezoneSource;
  // Populated ONLY when `timezone_source === 'location'`. The agent
  // can read these to phrase `here` answers — without them the agent
  // doesn't know coords and should say so.
  location_lat?: number;
  location_lng?: number;
  location_age_minutes?: number;
}

export interface BuildAgentContextInput {
  now: Date;
  /** `tz_state.home_tz` — fallback when walker has no covering segment. */
  homeTimezone: string;
  /** Container's configured `TZ` env. Used only when the resolver doesn't run. */
  containerTimezone: string;
  /** Owner's most-recent location row (post-#574). `null` when none recorded. */
  latestLocation: LocationRecord | null;
  /**
   * Decoded `tz_state.segments` payload. `null` when the column is
   * empty / NULL / malformed JSON. Empty array means "synced but no
   * segments" and is treated the same as null for resolver purposes.
   */
  segments: readonly TripitSegment[] | null;
}

export function buildAgentContext(input: BuildAgentContextInput): AgentContext {
  const { now, homeTimezone, containerTimezone, latestLocation, segments } =
    input;

  const segmentsForResolver: readonly TripitSegment[] | null =
    segments && segments.length > 0 ? segments : null;

  let timezone: string;
  let timezoneSource: AgentContextTimezoneSource;
  let locationFields: Pick<
    AgentContext,
    'location_lat' | 'location_lng' | 'location_age_minutes'
  > = {};

  // Match `runTzHeartbeatAdvisory`'s "no usable input" early-return:
  // when there's NO location AND NO segments, the resolver would
  // silently return `home_tz` via the walker — but for `<context>`
  // the user-visible meaning is "we have no signal", which is
  // `container_default`, NOT "you're at home".
  if (latestLocation === null && segmentsForResolver === null) {
    timezone = resolveTimezone(containerTimezone);
    timezoneSource = 'container_default';
  } else {
    const resolved = resolveCurrentTz({
      now,
      latestLocation,
      segments: segmentsForResolver,
      home_tz: homeTimezone,
    });
    timezone = resolved.tz;
    timezoneSource = mapResolverSource(
      resolved.source,
      resolved.tz,
      homeTimezone,
      segmentsForResolver,
      now,
    );
    if (
      timezoneSource === 'location' &&
      latestLocation &&
      resolved.latest_location_age_seconds !== null
    ) {
      locationFields = {
        location_lat: latestLocation.latitude,
        location_lng: latestLocation.longitude,
        location_age_minutes: Math.round(
          resolved.latest_location_age_seconds / 60,
        ),
      };
    }
  }

  return {
    utc_datetime: isoUtcNoMillis(now),
    local_datetime: isoLocalWithOffset(now, timezone),
    local_date: localDateOnly(now, timezone),
    weekday: formatWeekday(now, timezone),
    timezone,
    timezone_source: timezoneSource,
    ...locationFields,
  };
}

/**
 * Render the AgentContext as the `<context ... />` self-closing tag
 * that prefixes every agent prompt. Attribute order follows the
 * `#576` issue body for human readability; XML doesn't care about
 * attribute order so consumers MUST NOT depend on it. Numbers are
 * stringified without trailing zeros via `String(n)`.
 */
/**
 * Convenience wrapper for the orchestrator's hot path: read the
 * required inputs from SQLite (one tz_state read + one locations
 * read), build the AgentContext, and format it as the `<context />`
 * tag string ready to drop into `formatMessages`.
 *
 * Returns the formatted tag string. The orchestrator passes this
 * through to `formatMessages` as `MessageFormatContext.contextTag`
 * and uses the same `AgentContext.timezone` as the per-message
 * `time=` zone so the two are guaranteed consistent.
 *
 * `ownerSenderId` may be null/empty when `ASSISTANT_OWNER_TG_USER_ID`
 * isn't configured — the location-first cascade is skipped and the
 * cascade falls through to walker / container default. Same
 * fail-open contract as `runTzHeartbeatAdvisory`.
 */
export function buildAgentContextFromDb(args: {
  now?: Date;
  ownerSenderId: string | null | undefined;
  containerTimezone: string;
}): AgentContext {
  const now = args.now ?? new Date();
  const tzState = readTzStateForContext();
  const latestLocation = args.ownerSenderId
    ? getLatestLocationForSender(args.ownerSenderId)
    : null;

  // When `tz_state` is missing or at an unfamiliar schema version,
  // we have no `home_tz` and the resolver can't run — fall through
  // to container_default. Same contract as `runTzHeartbeatAdvisory`'s
  // schema gate.
  if (tzState === null) {
    return buildAgentContext({
      now,
      homeTimezone: args.containerTimezone,
      containerTimezone: args.containerTimezone,
      latestLocation: null,
      segments: null,
    });
  }

  return buildAgentContext({
    now,
    homeTimezone: tzState.home_tz,
    containerTimezone: args.containerTimezone,
    latestLocation,
    segments: tzState.segments,
  });
}

export function formatAgentContextTag(ctx: AgentContext): string {
  const parts: string[] = [
    `utc_datetime="${escapeXml(ctx.utc_datetime)}"`,
    `local_datetime="${escapeXml(ctx.local_datetime)}"`,
    `local_date="${escapeXml(ctx.local_date)}"`,
    `weekday="${escapeXml(ctx.weekday)}"`,
    `timezone="${escapeXml(ctx.timezone)}"`,
    `timezone_source="${escapeXml(ctx.timezone_source)}"`,
  ];
  if (ctx.location_lat !== undefined && ctx.location_lng !== undefined) {
    parts.push(`location_lat="${String(ctx.location_lat)}"`);
    parts.push(`location_lng="${String(ctx.location_lng)}"`);
  }
  if (ctx.location_age_minutes !== undefined) {
    parts.push(`location_age_minutes="${String(ctx.location_age_minutes)}"`);
  }
  return `<context ${parts.join(' ')} />`;
}

function mapResolverSource(
  resolverSource: TzResolverSource,
  resolvedTz: string,
  homeTz: string,
  segments: readonly TripitSegment[] | null,
  now: Date,
): AgentContextTimezoneSource {
  if (resolverSource === 'fresh_location') return 'location';
  // All walker_* paths: distinguish whether the walker hit a covering
  // segment (→ `segment`) or fell back to home_tz (→ `home_fallback`).
  // The walker doesn't return its branch, so re-derive: if running
  // the walker over the given segments produces home_tz, the fallback
  // path fired; otherwise a segment drove the answer. This is the
  // pre-Phase-2 walker contract and is stable.
  if (segments === null) return 'home_fallback';
  const walkerOutput = walkTzSegments(segments, now, homeTz);
  if (walkerOutput !== resolvedTz) {
    // Defensive: the resolver and our re-derivation disagree. Treat
    // as `segment` since the resolver's answer is authoritative and a
    // non-home tz must have come from somewhere. Should never fire in
    // practice; surfacing as `segment` avoids mislabeling the agent's
    // anchor.
    return 'segment';
  }
  return walkerOutput === homeTz ? 'home_fallback' : 'segment';
}

function isoUtcNoMillis(d: Date): string {
  return d.toISOString().replace(/\.\d{3}Z$/, 'Z');
}

function isoLocalWithOffset(d: Date, tz: string): string {
  const dtf = new Intl.DateTimeFormat('en-US', {
    timeZone: resolveTimezone(tz),
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    timeZoneName: 'longOffset',
  });
  const partsArr = dtf.formatToParts(d);
  const parts: Record<string, string> = {};
  for (const p of partsArr) {
    parts[p.type] = p.value;
  }
  // Intl can emit "24" for midnight in some locales; normalise to "00".
  const hour = parts.hour === '24' ? '00' : parts.hour;
  const offset = formatIntlOffset(parts.timeZoneName ?? 'GMT');
  return `${parts.year}-${parts.month}-${parts.day}T${hour}:${parts.minute}:${parts.second}${offset}`;
}

function localDateOnly(d: Date, tz: string): string {
  const dtf = new Intl.DateTimeFormat('en-CA', {
    timeZone: resolveTimezone(tz),
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
  // en-CA produces YYYY-MM-DD directly.
  return dtf.format(d);
}

function formatWeekday(d: Date, tz: string): string {
  return new Intl.DateTimeFormat('en-US', {
    timeZone: resolveTimezone(tz),
    weekday: 'long',
  }).format(d);
}

function formatIntlOffset(intlOffset: string): string {
  // Intl's `longOffset` returns "GMT" for exact UTC and "GMT±HH:MM"
  // otherwise. We want ISO-8601's "Z" for UTC and "±HH:MM" elsewhere
  // so `local_datetime` parses cleanly with `new Date(...)`.
  if (
    intlOffset === 'GMT' ||
    intlOffset === 'GMT+00:00' ||
    intlOffset === 'GMT-00:00'
  ) {
    return 'Z';
  }
  return intlOffset.replace(/^GMT/, '');
}
