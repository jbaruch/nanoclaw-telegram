# Precheck-Gating Audit Replay (#375)

Replay procedure for the [precheck-gating audit](precheck-gating-audit.md) committed at `2026-05-01T02:00Z`. Re-runs quarterly so new recurring tasks added between snapshots can't slip in without prechecks (the failure mode #338 catalogued — 5/6 tasks were missing one at first audit).

## When to run

Quarterly. The methodology is window-agnostic but the script defaults to a 90-day window (cron-weekly tasks need ≥4 fires for empirical confidence on duration drift).

## Operator one-liner

Register the recurring agent with `/schedule` once. Cadence: `0 6 1 1,4,7,10 *` (06:00 local on the 1st of January, April, July, October — quarterly).

```
/schedule cron='0 6 1 1,4,7,10 *' prompt='Run the procedure in docs/precheck-gating-audit-replay.md against jbaruch/nanoclaw and open a PR per the threshold rule.'
```

The `/schedule` skill requires claude.ai authentication (not API). Register from a CLI session that has run `/login` against claude.ai.

## Procedure (the agent follows this)

Process steps in order. Do not skip ahead.

### Step 1 — Snapshot the live state

Run the deterministic snapshotter against the NAS DB. From the `nanoclaw` checkout root:

```bash
ssh nas 'cd ~/nanoclaw && npx tsx scripts/audit-precheck-gating.ts --db store/messages.db --days 90' > /tmp/snapshot.json
```

The output is a single JSON object with `snapshot_at`, `window_start`/`window_end`, `window_days`, and `tasks: [...]`. Each task carries `task_id`, `group_folder`, `schedule_type`, `schedule_value`, `has_precheck`, `script`, and run-log stats (`fires`, `gated_likely`, `avg_duration_s`, `min_duration_s`, `max_duration_s`).

### Step 2 — Diff against the prior audit

Read `docs/precheck-gating-audit.md` and compare its catalogue table to the snapshot:

- **Added tasks** — `task_id` in snapshot but not in the prior catalogue.
- **Removed tasks** — `task_id` in the prior catalogue but not in the snapshot (cancelled, deleted, or moved to a once-task).
- **Drift** — `task_id` in both; flag any of:
  - `has_precheck` flipped (a task lost or gained a precheck)
  - `avg_duration_s` shifted by ≥ 25 % vs. the prior catalogue's `Avg duration`
  - `min_duration_s` floor crossed the 10 s boundary in either direction (suggests a precheck-gate appeared or disappeared)

### Step 3 — Pick the output mode by diff size

Count of `added + removed + drift` tasks:

- **0 changes** — no PR. Comment on `#375`'s parent epic with `unchanged at <snapshot_at>` and exit silently.
- **1–2 changes** — open a PR updating `docs/precheck-gating-audit.md` in place. Update the catalogue table row(s), the per-task analysis section, and the "Recommended follow-ups" list if any new tasks are missing prechecks. Title: `chore(audit): refresh precheck-gating audit (#375 quarterly Q<N>-YYYY)`.
- **≥ 3 changes** — file a NEW dated audit doc at `docs/precheck-gating-audit-YYYY-Q<N>.md`. Don't touch the original — it stays as the historical record. Title: `chore(audit): file fresh precheck-gating audit (#375 quarterly Q<N>-YYYY)`.

### Step 4 — File any new follow-up issues

For every "added task" without a precheck, file a follow-up issue under the `enhancement` label naming the task and recommending a precheck shape per the methodology in the original audit's "Per-task analysis" sections. Reference `#338` and `#375`.

If a "drift" finding shows a task that *had* a precheck losing it (or its gate-out rate collapsing), file a regression issue tagged `bug`.

### Step 5 — Open the PR + tag for review

`gh pr create --repo jbaruch/nanoclaw` with the diff against `main`. Body must explicitly call out:

- Snapshot timestamp + window
- Counts: added / removed / drift / unchanged
- Output mode chosen (in-place vs. new doc)
- Links to any follow-up issues filed in Step 4

Watch CI to completion per `jbaruch/coding-policy: ci-safety.md`. The audit doc is pure prose so the only checks are policy reviewers.

### Step 6 — Finish

If Step 3 chose the "0 changes" branch, exit silently. Otherwise, the PR is the deliverable; merge follows the normal review-and-merge cadence.

## Why this lives outside `/schedule`'s prompt

`/schedule` registers a remote agent that runs the full prompt verbatim each quarter. Inlining the entire procedure into the cron registration means every minor methodology refinement requires a re-registration and a new agent identity. Keeping the procedure here lets the cron prompt stay one short instruction (`follow docs/precheck-gating-audit-replay.md, open a PR`) while the methodology evolves in-tree under normal PR review.

## Refs

- Parent: `jbaruch/nanoclaw#338`
- This task: `jbaruch/nanoclaw#375`
- Audit doc: [`precheck-gating-audit.md`](precheck-gating-audit.md)
- Snapshotter: [`scripts/audit-precheck-gating.ts`](../scripts/audit-precheck-gating.ts) (core in [`src/audit-precheck-gating.ts`](../src/audit-precheck-gating.ts))
