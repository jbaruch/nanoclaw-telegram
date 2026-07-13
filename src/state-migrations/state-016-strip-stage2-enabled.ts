import type { StateMigration } from '../db.js';

/**
 * #753 — strip the retired `container_config.stage2Enabled` flag from
 * every `registered_groups` row.
 *
 * `stage2Enabled` backed the Stage-2 Haiku classifier subsystem, removed
 * from the codebase during the subscription-OAuth cutover. No production
 * reader references it anymore and the `ContainerConfig` type no longer
 * declares it, but the field lingers in stored `container_config` blobs
 * (main was `false`; several trusted groups `true`). It confuses operators
 * and agents reading DB dumps and implies a feature that no longer exists.
 *
 * `parseContainerConfig` round-trips unknown keys verbatim, so a plain
 * read-modify-write never drops the dead field — it has to be excised from
 * the stored JSON directly. `json_remove` rewrites each blob without the
 * key; the write-path guard in `setRegisteredGroup` (added alongside this
 * migration) keeps it from coming back via a hand-edited row or a stale
 * IPC config blob.
 *
 * Conservative WHERE clause: only rows whose config is valid JSON AND
 * actually carry the key are touched — a NULL config, a (corrupt)
 * non-JSON blob, or an already-clean row is left byte-identical rather
 * than needlessly rewritten. The valid-JSON check and the key lookup are
 * ordered by a `CASE` rather than chained `AND` terms: SQLite's optimizer
 * may reorder `WHERE` conjuncts, so a bare `json_valid(...) AND
 * json_type(...)` could evaluate the lookup first and raise "malformed
 * JSON" on a corrupt blob, aborting startup instead of skipping the row.
 * `CASE` guarantees the guard runs first; a corrupt blob short-circuits to
 * `0` and is left untouched.
 *
 * Key PRESENCE is tested with `json_type`, not `json_extract(...) IS NOT
 * NULL`: `json_extract` returns SQL NULL both for an absent key AND for a
 * key whose value is JSON `null`, so the latter would slip through
 * unstripped. `json_type(container_config, '$.stage2Enabled')` returns the
 * value's type when the key exists (any value, `null` included) and SQL
 * NULL only when the key is absent.
 *
 * No `schema_version` bump: `registered_groups` carries no per-row schema
 * version to advance, and dropping an ignored key is backward-compatible —
 * an older binary reading a scrubbed row simply sees no `stage2Enabled`,
 * exactly as it already treated the field (unread).
 */
export const STATE_016_STRIP_STAGE2_ENABLED: StateMigration = {
  version: 16,
  name: 'strip retired container_config.stage2Enabled from registered_groups (#753)',
  sql: `
    UPDATE registered_groups
    SET container_config = json_remove(container_config, '$.stage2Enabled')
    WHERE container_config IS NOT NULL
      AND CASE
            WHEN json_valid(container_config)
              THEN json_type(container_config, '$.stage2Enabled') IS NOT NULL
            ELSE 0
          END;
  `,
};
