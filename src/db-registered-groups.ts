// Registered-group accessors (#751 seam 4, extracted verbatim from
// src/db.ts): the `registered_groups` table — group registration,
// trust flips, trigger-pattern config (#81/#82), container config.
// Reads the shared connection's live binding from `db-connection.ts`.
import { db } from './db-connection.js';
import { isValidGroupFolder } from './group-folder.js';
import { logger } from './logger.js';
import {
  ContainerConfig,
  RegisteredGroup,
  TriggerPattern,
  TriggerPatternConfig,
} from './types.js';

// Defensive parser shared by getRegisteredGroup and getAllRegisteredGroups.
// A single malformed row (partial write, manual edit, schema-migration glitch)
// must not crash startup — getAllRegisteredGroups runs at orchestrator boot.
//
// Catches SyntaxError specifically (JSON.parse's only throw); other errors
// propagate. Validates the parsed value is a non-null object — JSON.parse
// can legally return primitives, null, or arrays from `"null"`, `"true"`,
// `"[]"`, etc., none of which are valid ContainerConfig shapes.
//
// Logs jid + payload length only — never the payload content. The raw
// container_config string is treated as opaque/possibly-sensitive per
// no-secrets and error-handling rules. Operators inspect the actual row
// via the DB by jid, not via logs.
function parseContainerConfig(
  raw: string | null,
  jid: string,
): ContainerConfig | undefined {
  // Distinguish SQL NULL from empty string: NULL is the documented
  // "no config" state, while an empty string in a TEXT column is itself
  // a corruption indicator (something wrote "" where it should have
  // written NULL). Fall through into the parse path so SyntaxError
  // surfaces it.
  if (raw === null) return undefined;
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
    logger.warn(
      { errName: err.name, jid, len: raw.length },
      'registered_groups: invalid container_config JSON, treating as undefined',
    );
    return undefined;
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    logger.warn(
      {
        jid,
        len: raw.length,
        parsedType:
          parsed === null
            ? 'null'
            : Array.isArray(parsed)
              ? 'array'
              : typeof parsed,
      },
      'registered_groups: container_config is not a JSON object, treating as undefined',
    );
    return undefined;
  }
  return parsed as ContainerConfig;
}

// Dual-mode reader for `registered_groups.trigger_pattern` (#81).
// Pre-#81 rows store a single string ("@Andy"). Post-#81 rows store a
// JSON-encoded `TriggerPatternConfig`. The createSchema-time backfill
// converts every legacy row to JSON on first boot, but we keep the
// dual-mode read path for two reasons:
//   1. A user rolling back to a pre-#81 binary then forward again must
//      re-converge cleanly without manual intervention.
//   2. The setup/register CLI and external tooling sometimes write
//      raw strings directly via `_writeRawRegisteredGroup` for tests;
//      we want those to keep working without forcing every test to
//      know the JSON shape.
//
// Returns the canonical config (or null when the row is unreadable)
// plus the derived trigger string used to populate
// `RegisteredGroup.trigger`. Three failure modes are distinguished:
//
//   - Legacy string shape (no leading `{`) → wrap as a synthesised
//     single-element keyword config. Reader-side compat for rollback
//     and direct test writes.
//   - JSON parse fails or shape doesn't validate → loud warn, fall
//     back to a legacy keyword config so a single bad row can't
//     crash boot. The trigger string surfaces the raw column so
//     operators can still see what's there.
//   - JSON parses, shape validates, BUT version > 1 (a future
//     binary's row read by this older binary) → loud warn, return
//     `null` config and `null` trigger. The gate framework treats
//     null as "no opinion" and falls through to the next gate, so
//     the failure is loud-but-non-fatal. See review of #84,
//     comment-id 4360940433: silent fallback-to-keyword on a future
//     row was too quiet.
function parseTriggerPatternColumn(
  raw: string,
  jid: string,
): { config: TriggerPatternConfig | null; trigger: string | null } {
  const trimmed = raw.trim();
  // Cheap shape sniff: JSON config always starts with `{`. Anything
  // else is the legacy string shape (or corruption that we treat as
  // legacy by best-effort).
  if (!trimmed.startsWith('{')) {
    const legacy = legacyTriggerToConfig(raw);
    return {
      config: legacy,
      trigger: deriveTriggerString(legacy),
    };
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
    logger.warn(
      { errName: err.name, jid, len: raw.length },
      'registered_groups: invalid trigger_pattern JSON, falling back to legacy keyword shape',
    );
    const legacy = legacyTriggerToConfig(raw);
    return {
      config: legacy,
      trigger: raw,
    };
  }
  // Detect a future-version row BEFORE the strict-shape validator
  // would also reject it: a `{version: 2, ...}` row read by a v1
  // binary is a forward-compat scenario (rollback), not corruption.
  // Surface it loudly so the operator notices, and return null so
  // downstream gates treat the row as "no patterns configured"
  // (== fail-open at the gate combinator) rather than silently
  // re-interpreting the raw JSON as a literal keyword pattern.
  if (parsed !== null && typeof parsed === 'object' && !Array.isArray(parsed)) {
    const v = (parsed as Record<string, unknown>).version;
    if (typeof v === 'number' && v > 1) {
      logger.warn(
        { jid, foundVersion: v, expectedVersion: 1, len: raw.length },
        'registered_groups: trigger_pattern JSON has unsupported future version, treating as no-config (gates will fall through to fail-open)',
      );
      return { config: null, trigger: null };
    }
  }
  if (!isTriggerPatternConfig(parsed)) {
    logger.warn(
      { jid, len: raw.length },
      'registered_groups: trigger_pattern JSON is not a valid TriggerPatternConfig, falling back to legacy keyword shape',
    );
    const legacy = legacyTriggerToConfig(raw);
    return {
      config: legacy,
      trigger: raw,
    };
  }
  return {
    config: parsed,
    trigger: deriveTriggerString(parsed),
  };
}

function legacyTriggerToConfig(raw: string): TriggerPatternConfig {
  return {
    version: 1,
    patterns: [
      {
        pattern: raw,
        kind: 'keyword',
        source: 'owner-set',
        precision: 0,
        sample_count: 0,
        last_matched_at: null,
        last_updated_at: null,
      },
    ],
  };
}

/**
 * Derive the legacy `RegisteredGroup.trigger` string from a parsed
 * config. Pinned-down replacement for the unspecified
 * "first-keyword-or-empty" behaviour flagged on PR #84
 * (comment-id 4360940433).
 *
 * Order of preference, locked down so #82 can write mention-only /
 * regex-only configs without surprising any legacy reader:
 *   1. First `keyword`-kind pattern → return its `pattern` verbatim.
 *      This preserves the pre-#81 shape where `@Andy` was stored as
 *      a literal `@Andy` keyword.
 *   2. Else first `mention`-kind pattern → return `'@' + pattern`.
 *      Mentions are stored bare per the migration backfill (Fix 1
 *      above), so re-prepending the `@` keeps the legacy regex
 *      `(?:^|\s)@Andy\b` alive for call sites that haven't migrated
 *      to the new gate.
 *   3. Else `null`. Downstream call sites coerce null → undefined
 *      and let `getTriggerPattern` fall back to the global default
 *      `@<assistant>` regex; this preserves message-loop wakeup for
 *      the global mention even on a config with no per-group
 *      keyword/mention. Returning `null` (vs the previous empty
 *      string) is the type-system signal that "there is no
 *      group-specific trigger string" — ambiguous before, explicit
 *      now.
 */
export function deriveTriggerString(
  config: TriggerPatternConfig | null,
): string | null {
  if (!config) return null;
  const keyword = config.patterns.find((p) => p.kind === 'keyword');
  if (keyword) return keyword.pattern;
  const mention = config.patterns.find((p) => p.kind === 'mention');
  if (mention) {
    return mention.pattern.startsWith('@')
      ? mention.pattern
      : `@${mention.pattern}`;
  }
  return null;
}

// Allow-listed enums for `kind` / `source` per the TriggerPatternKind /
// TriggerPatternSource union types in src/types.ts. Validation here means
// a typo like `kind: "mentoin"` is caught at parse time and the row falls
// through to the warn-log path in `parseTriggerPatternColumn`, instead of
// silently being treated as "valid config" and then ignored downstream
// because `deriveTriggerString` finds no `keyword`/`mention` it recognises.
const TRIGGER_PATTERN_KINDS: ReadonlySet<string> = new Set([
  'keyword',
  'mention',
  'reply',
  'regex',
  'sender_tier',
]);
const TRIGGER_PATTERN_SOURCES: ReadonlySet<string> = new Set([
  'owner-set',
  'learned',
  'universal',
]);

// Exported for the legacy trigger_pattern schema migration in `db.ts`
// (createSchema #81 backfill); internal shape guard otherwise.
export function isTriggerPatternConfig(
  value: unknown,
): value is TriggerPatternConfig {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    return false;
  }
  const v = value as Record<string, unknown>;
  if (v.version !== 1) return false;
  if (!Array.isArray(v.patterns)) return false;
  for (const p of v.patterns) {
    if (p === null || typeof p !== 'object' || Array.isArray(p)) return false;
    const pp = p as Record<string, unknown>;
    if (typeof pp.pattern !== 'string') return false;
    if (typeof pp.kind !== 'string' || !TRIGGER_PATTERN_KINDS.has(pp.kind)) {
      return false;
    }
    if (
      typeof pp.source !== 'string' ||
      !TRIGGER_PATTERN_SOURCES.has(pp.source)
    ) {
      return false;
    }
    if (typeof pp.precision !== 'number') return false;
    if (typeof pp.sample_count !== 'number') return false;
    if (pp.last_matched_at !== null && typeof pp.last_matched_at !== 'string') {
      return false;
    }
    if (pp.last_updated_at !== null && typeof pp.last_updated_at !== 'string') {
      return false;
    }
  }
  return true;
}

/**
 * Serialise a `RegisteredGroup` into the column value for
 * `trigger_pattern`. Always emits JSON (forward-only writes). When the
 * caller already has a `triggerPatterns` config, it wins; otherwise
 * we synthesise a single-element keyword config from `group.trigger`
 * with `source: "owner-set"` so the row is observability-ready
 * immediately. A null `group.trigger` with no `triggerPatterns`
 * yields an empty-pattern config — readable by the gate framework as
 * "no patterns configured" (== `pass`, fail-open).
 */
function serializeTriggerPatternForColumn(group: RegisteredGroup): string {
  if (group.triggerPatterns) return JSON.stringify(group.triggerPatterns);
  if (group.trigger === null || group.trigger === undefined) {
    return JSON.stringify({ version: 1, patterns: [] });
  }
  return JSON.stringify(legacyTriggerToConfig(group.trigger));
}

/**
 * Read the full trigger pattern config for a group.
 *
 * Returns:
 *   - `undefined` — the group does not exist (no row).
 *   - `null` — the row exists but is unreadable for the current
 *     binary (e.g. JSON parse OK but `version > 1`, signalling a
 *     forward-binary write read by an older binary). Loud-warned at
 *     parse time. The trigger gate consumes null as "no opinion"
 *     (decision: `pass`), which falls through the chain to default-
 *     allow rather than silent-misinterpret. See review of #84,
 *     comment-id 4360940433.
 *   - `TriggerPatternConfig` otherwise. Legacy string rows surface as
 *     a synthesised single-element config.
 */
export function getTriggerPatterns(
  jid: string,
): TriggerPatternConfig | null | undefined {
  const row = db
    .prepare('SELECT trigger_pattern FROM registered_groups WHERE jid = ?')
    .get(jid) as { trigger_pattern: string } | undefined;
  if (!row) return undefined;
  return parseTriggerPatternColumn(row.trigger_pattern, jid).config;
}

/**
 * Replace the trigger pattern config for an existing group. Throws
 * if the group is not registered (callers should `setRegisteredGroup`
 * first). Updates `trigger_pattern` only — other columns untouched.
 */
export function setTriggerPatterns(
  jid: string,
  config: TriggerPatternConfig,
): void {
  if (config.version !== 1) {
    throw new Error(
      `Unsupported TriggerPatternConfig version ${config.version} for jid ${jid}`,
    );
  }
  const result = db
    .prepare(`UPDATE registered_groups SET trigger_pattern = ? WHERE jid = ?`)
    .run(JSON.stringify(config), jid);
  if (result.changes === 0) {
    throw new Error(
      `setTriggerPatterns: no registered_groups row for jid ${jid}`,
    );
  }
}

/**
 * Re-export the row-level `TriggerPattern` type so #82's self-improvement
 * code can import its observability shape from a single place.
 */
export type { TriggerPattern, TriggerPatternConfig };

export function getRegisteredGroup(
  jid: string,
): (RegisteredGroup & { jid: string }) | undefined {
  const row = db
    .prepare('SELECT * FROM registered_groups WHERE jid = ?')
    .get(jid) as
    | {
        jid: string;
        name: string;
        folder: string;
        trigger_pattern: string;
        added_at: string;
        container_config: string | null;
        requires_trigger: number | null;
        is_main: number | null;
      }
    | undefined;
  if (!row) return undefined;
  if (!isValidGroupFolder(row.folder)) {
    logger.warn(
      { jid: row.jid, folder: row.folder },
      'Skipping registered group with invalid folder',
    );
    return undefined;
  }
  const triggerParsed = parseTriggerPatternColumn(row.trigger_pattern, row.jid);
  return {
    jid: row.jid,
    name: row.name,
    folder: row.folder,
    trigger: triggerParsed.trigger,
    added_at: row.added_at,
    containerConfig: parseContainerConfig(row.container_config, row.jid),
    requiresTrigger:
      row.requires_trigger === null ? undefined : row.requires_trigger === 1,
    isMain: row.is_main === 1 ? true : undefined,
    triggerPatterns: triggerParsed.config ?? undefined,
  };
}

/**
 * Retired `container_config` keys stripped on every write (#753).
 *
 * `stage2Enabled` backed the removed Stage-2 Haiku classifier. The
 * `ContainerConfig` type no longer declares it and no reader consults
 * it, but `parseContainerConfig` round-trips unknown keys verbatim — so
 * a read-modify-write of a legacy row, or an IPC config blob from an
 * older client, would re-persist the dead field that migration
 * `state-016` scrubbed from stored blobs. `setRegisteredGroup` is the
 * single write chokepoint for the row; stripping here closes the
 * reintroduction path in one place.
 */
const RETIRED_CONTAINER_CONFIG_KEYS = ['stage2Enabled'] as const;

/**
 * Return a `container_config` JSON string with every retired key removed,
 * or `null` when there is no config. Operates on a shallow copy so the
 * caller's `ContainerConfig` object is never mutated. Only allocates a
 * copy when a retired key is actually present.
 */
function serializeContainerConfigForColumn(
  containerConfig: ContainerConfig | undefined,
): string | null {
  if (!containerConfig) return null;
  const present = RETIRED_CONTAINER_CONFIG_KEYS.filter(
    (key) => key in containerConfig,
  );
  if (present.length === 0) return JSON.stringify(containerConfig);
  const cleaned = { ...containerConfig } as Record<string, unknown>;
  for (const key of present) delete cleaned[key];
  return JSON.stringify(cleaned);
}

export function setRegisteredGroup(jid: string, group: RegisteredGroup): void {
  if (!isValidGroupFolder(group.folder)) {
    throw new Error(`Invalid group folder "${group.folder}" for JID ${jid}`);
  }
  db.prepare(
    `INSERT OR REPLACE INTO registered_groups (jid, name, folder, trigger_pattern, added_at, container_config, requires_trigger, is_main)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
  ).run(
    jid,
    group.name,
    group.folder,
    serializeTriggerPatternForColumn(group),
    group.added_at,
    serializeContainerConfigForColumn(group.containerConfig),
    // Map TS `undefined` to SQL NULL (not 0). NULL and 0 are distinct
    // states elsewhere in the orchestrator: `index.ts` checks
    // `requiresTrigger === false` to decide whether to skip a group's
    // heartbeat sync, and a NULL row should NOT match that branch.
    // Pre-#105, this column wrote 0 for undefined, which silently
    // collapsed the NULL state on every round-trip — biting the new
    // partial-update helpers (`updateGroupTrusted`/`updateGroupTrigger`)
    // because they read existing → reapply. Callers that want explicit
    // false must pass `false` explicitly; callers passing `undefined`
    // get NULL preserved.
    group.requiresTrigger === undefined ? null : group.requiresTrigger ? 1 : 0,
    group.isMain ? 1 : 0,
  );
}

/**
 * Partial update: flip `containerConfig.trusted` only.
 *
 * Returns the updated RegisteredGroup, or `undefined` if the JID isn't
 * registered. The caller is responsible for refreshing in-memory state
 * and snapshots — this function only touches the DB row.
 *
 * Implementation note: we round-trip through `getRegisteredGroup` to
 * preserve every other field (additionalMounts, isMain, etc.) verbatim,
 * then write back via `setRegisteredGroup`. A targeted SQL UPDATE on the
 * JSON column would be marginally faster but would force us to either
 * mutate the JSON string textually (fragile) or duplicate the JSON
 * encoding logic that already lives in `setRegisteredGroup`.
 */
export function updateGroupTrusted(
  jid: string,
  trusted: boolean,
): RegisteredGroup | undefined {
  const existing = getRegisteredGroup(jid);
  if (!existing) return undefined;
  // `getRegisteredGroup` synthesizes its return value with `jid` set as
  // an extra runtime field for caller convenience, but `RegisteredGroup`
  // doesn't declare it. Strip via destructure before spreading so the
  // value we hand back (and the in-memory cache the orchestrator
  // mirrors into) doesn't carry the DB-only key.
  const { jid: _existingJid, ...rest } = existing;
  void _existingJid;
  const updated: RegisteredGroup = {
    ...rest,
    containerConfig: {
      ...(rest.containerConfig ?? {}),
      trusted,
    },
  };
  setRegisteredGroup(jid, updated);
  return updated;
}

/**
 * Partial update: change `trigger_pattern` and optionally `requires_trigger`
 * only. Other fields preserved. Returns updated group or `undefined` if
 * the JID isn't registered or the trigger fails the non-empty invariant.
 *
 * Why reject empty/whitespace triggers: `getTriggerPattern('')` trims
 * and falls back to `DEFAULT_TRIGGER`, so a caller that thinks they're
 * setting a custom trigger would silently get the assistant's default
 * trigger word instead — not what they asked for. Reject at the DB
 * boundary so any future caller (cron migrations, manual fixups,
 * alternate MCP tools) can't bypass the IPC-layer check.
 *
 * The trigger is also `.trim()`ed before persistence so `' @Andy '`
 * doesn't end up stored with surrounding whitespace (which would render
 * that way in `available_groups.json` and elsewhere).
 */
export function updateGroupTrigger(
  jid: string,
  trigger: string,
  requiresTrigger?: boolean,
): RegisteredGroup | undefined {
  if (typeof trigger !== 'string' || trigger.trim().length === 0) {
    logger.warn(
      { jid },
      'updateGroupTrigger: rejecting empty/whitespace trigger',
    );
    return undefined;
  }
  const normalizedTrigger = trigger.trim();
  const existing = getRegisteredGroup(jid);
  if (!existing) return undefined;
  // Strip the DB-only `jid` field so it doesn't leak into the returned
  // RegisteredGroup or into the in-memory cache the orchestrator
  // mirrors into. Same rationale as updateGroupTrusted above.
  const {
    jid: _existingJid,
    triggerPatterns: _existingTriggerPatterns,
    ...rest
  } = existing;
  void _existingJid;
  void _existingTriggerPatterns;
  const updated: RegisteredGroup = {
    ...rest,
    trigger: normalizedTrigger,
    ...(requiresTrigger === undefined ? {} : { requiresTrigger }),
  };
  setRegisteredGroup(jid, updated);
  // `triggerPatterns` was stripped above so the serializer re-derives the
  // column from the new `trigger`. Re-read the persisted config back into
  // the returned object: callers mirror this into the in-memory registry
  // the gate reads, and an object with `triggerPatterns: undefined` would
  // leave the trigger gate fail-open until the next reload (#670).
  return { ...updated, triggerPatterns: getTriggerPatterns(jid) ?? undefined };
}

/**
 * Remove a registered_groups row by JID. Returns true if a row was
 * actually deleted, false if no row matched. Idempotent — repeat calls
 * after deletion are a no-op and report `false`.
 *
 * Caller is responsible for refreshing in-memory state and snapshots —
 * this function only touches the DB row, mirroring the
 * `setRegisteredGroup` / `updateGroupTrusted` contract.
 *
 * Out of scope: the on-disk `groups/<folder>/` directory. Group state
 * (CLAUDE.md, MEMORY.md, scheduled-task workspace) survives unregister
 * — operators delete those manually if/when they want a clean slate.
 * Forces a deliberate destructive action instead of silently nuking
 * agent-curated state when the registration churns.
 */
export function deleteRegisteredGroup(jid: string): boolean {
  const result = db
    .prepare('DELETE FROM registered_groups WHERE jid = ?')
    .run(jid);
  return result.changes > 0;
}

export function getAllRegisteredGroups(): Record<string, RegisteredGroup> {
  const rows = db.prepare('SELECT * FROM registered_groups').all() as Array<{
    jid: string;
    name: string;
    folder: string;
    trigger_pattern: string;
    added_at: string;
    container_config: string | null;
    requires_trigger: number | null;
    is_main: number | null;
  }>;
  const result: Record<string, RegisteredGroup> = {};
  for (const row of rows) {
    if (!isValidGroupFolder(row.folder)) {
      logger.warn(
        { jid: row.jid, folder: row.folder },
        'Skipping registered group with invalid folder',
      );
      continue;
    }
    const triggerParsed = parseTriggerPatternColumn(
      row.trigger_pattern,
      row.jid,
    );
    result[row.jid] = {
      name: row.name,
      folder: row.folder,
      trigger: triggerParsed.trigger,
      added_at: row.added_at,
      containerConfig: parseContainerConfig(row.container_config, row.jid),
      requiresTrigger:
        row.requires_trigger === null ? undefined : row.requires_trigger === 1,
      isMain: row.is_main === 1 ? true : undefined,
      triggerPatterns: triggerParsed.config ?? undefined,
    };
  }
  return result;
}
