# Trigger Pattern Learner — Schema

Per `rules/stateful-artifacts.md`: the trigger-pattern learner persists
proposals into the existing `registered_groups.trigger_pattern` JSON
column (parsed as `TriggerPatternConfig` in `src/types.ts`). No new
table — the learner is a new owner of additional optional fields on
existing `TriggerPattern` records.

## Owner / writer / reader contract

- **Owner skill:** `src/gates/trigger-learner.ts` (this module).
  It is the SOLE writer of records with `source: 'learned'`. It also
  owns the optional `pattern_version`, `proposed_at`, `disabled`,
  `enabled`, `prior_versions` fields when they appear on a learned
  row.
- **Other writers:** the orchestrator's setup paths (`setRegisteredGroup`,
  `setTriggerPatterns`) write `source: 'owner-set'` rows. They do NOT
  write any of the learner-only optional fields.
- **Readers:** the `triggerGate` (`src/gates/trigger.ts`) reads every
  pattern in the array. It honours `enabled: false` (skip — pure
  proposal not yet promoted) and `disabled: true` (skip — auto-rolled
  back) by ignoring those rows entirely. Owner-set and universal
  patterns leave both flags unset and pass through unchanged.
- **Migration:** reader skills (gate matcher, admin tools) MUST tolerate
  a learned-row's optional fields being absent. Only the owner skill
  (the learner) writes them; `isTriggerPatternConfig` validates by
  required fields only and ignores unknown keys.

## Schema version

The wrapper config stays at `version: 1`. Adding optional fields is
backward-compatible: legacy rows pass the validator and existing
readers ignore the unknown keys. Bumping to `version: 2` would
trip `parseTriggerPatternColumn`'s "future version" warning and
disable the row. If a future change makes a field required, bump to
`version: 2` and provide a migration in this file.

## Optional fields owned by the learner

All fields below are OPTIONAL on `TriggerPattern` and only meaningful
for `source: 'learned'` rows. The owner skill is the only writer.

### `pattern_version: number`

Lineage version of THIS pattern body. Starts at `1` on first
proposal. The learner increments it each time it supersedes a
prior proposal for the same logical pattern (matched on
`{kind, pattern}` tuple). The owner can revert manually via the
admin promotion path by selecting the older record from
`prior_versions`.

### `proposed_at: string`

ISO-8601 timestamp the learner first proposed this pattern. Distinct
from `last_updated_at` (which advances on every metric refresh) and
`last_matched_at` (which advances on every match). `proposed_at`
records lineage — used by the dashboard / promotion UI to surface
"how long has this proposal been in the queue".

### `disabled: boolean`

`true` means auto-rolled-back: the learner's precision / FP-rate
window dropped below threshold and the loop demoted this row. The
record stays in the array (the owner may want to inspect why) but
the gate matcher ignores it. Default / unset / `false` means active.
Only the owner skill flips this field.

### `enabled: boolean`

`true` means the owner has promoted this proposal — the gate matcher
will consume it. Default / unset / `false` means pure proposal: the
learner has scored it as candidate but the owner hasn't approved it
yet. The trigger gate skips `enabled: false` rows. The promotion UI
flips this field; the learner itself does NOT (proposals start
inert).

### `prior_versions: TriggerPattern[]`

Snapshot of the immediately-prior version of this pattern, kept so
the owner can revert without losing prior precision metrics. The
learner caps this at one level deep — older history is dropped to
avoid unbounded column growth. The recursive type is fine because
inner records are always _prior_ states, never forward-pointers.

## Hints, not authority

The learner's proposals are HINTS, not ground truth. Per
`rules/stateful-artifacts.md`, the matcher (reader) does not blindly
trust the precision number on a row. Operator action (promotion via
`enabled: true`) is the authoritative gate; demotion via
`disabled: true` is the authoritative kill-switch. The matcher
honours both flags and otherwise ignores the precision metrics —
they're observability for the owner, not policy for the gate.
