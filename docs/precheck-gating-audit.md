# Precheck-Gating Audit (#338)

Audit of `wakeAgent` precheck-gating strictness across every active recurring scheduled task in production. Captured against the live `messages.db` on the NAS at `2026-05-01T02:00Z`; data window is the half-open range `[2026-04-17T02:00Z, 2026-05-01T02:00Z)` (14 days back from capture).

## Why this audit exists

Per [`jbaruch/coding-policy: rules/script-delegation.md` § Precheck Gating](https://github.com/jbaruch/coding-policy/blob/main/rules/script-delegation.md), recurring scheduled tasks should ship a precheck script that emits a JSON gate signal when there's nothing actionable. The agent-runner reads the script's last stdout line as JSON; on `wakeAgent: false` (or on any parse failure / script error) the LLM is never invoked, zero model tokens. (Note the rule documents the field as `wake_agent` (snake_case); the actual implementation in `container/agent-runner/src/index.ts` reads `wakeAgent` (camelCase). The audit recommendations below use `wakeAgent` to match the runtime contract; rule-vs-implementation alignment is a separate documentation fix outside #338's scope.) Gate-outs are pure profit — every fire that doesn't reach the agent is a 100% cost reduction for that fire, and gating compounds with every in-session optimisation downstream (#336 session reuse, #337 maintenance blocklist, #104 kill-auto-compaction).

The motivating cross-window finding from the cost-reduction effort: a meaningful fraction of recurring fires reached the agent despite having no actionable work — gating could have skipped them at zero token cost. This audit catalogues the "shouldn't have woken" cohort by inspecting every active recurring task's precheck script (or its absence) against the live data window.

## How the gate actually works

- Task row carries an optional `scheduled_tasks.script` column. When non-null its value is the **bash script content** the agent-runner runs **inside** the container before invoking the SDK — not a command line: the value is written verbatim to `/tmp/task-script.sh` and executed via `bash`. (Single-line script values like the heartbeat's `python3 /home/node/.claude/skills/.../heartbeat-precheck.py` happen to look like command lines, but the runtime treats them as shell content either way.)
- Mechanism: `src/task-scheduler.ts:500` passes `task.script` through to `ContainerInput`; `container/agent-runner/src/index.ts:2585-2591` writes it to `/tmp/task-script.sh` and `execFile('bash', ...)`s it; the gate at `container/agent-runner/src/index.ts:2837-2846` short-circuits both when `wakeAgent === false` AND when `scriptResult` is `null` (script error, timeout, or unparseable JSON output) — in either case the run returns `{ status: 'success', result: null }` without entering the SDK loop.
- When `wakeAgent: true`, the agent-runner prepends the script's `data` payload to the prompt as a `Script output:` block, so the script can pass deterministic data into the agent's first turn without a second fetch.
- Precheck-gated runs DO write a `task_run_logs` row (`status='success'`, `result=null`, `duration_ms` covering the full container lifetime including spawn + script + the post-result close delay). Operators can therefore identify gated fires by short `duration_ms` even though no LLM tokens were spent.

## Catalogue

Six active recurring tasks (`status='active' AND schedule_type IN ('cron', 'interval')`):

| Task id | Group | Schedule | Precheck | 14d fires | 14d gated | Avg duration (s) | Verdict |
|---|---|---|---|---|---|---|---|
| `task-1776347638739-jo99ce` | telegram_swarm | interval 30 min | `python3 .../tessl__heartbeat/scripts/heartbeat-precheck.py` | 882 | ~103 (12 %) — pre-2026-04-19 only | 549.1 | **Gate disabled by design** — see § Heartbeat below |
| `task-1776347641153-dexq4p` | telegram_swarm | cron `0 7 * * *` (daily 07:00) | NONE | 13 | 0 (0 %) | 711.2 | **Missing precheck** — fire-and-forget |
| `task-1776347643027-tlz3uc` | telegram_swarm | cron `0 3 * * *` (daily 03:00) | NONE | 13 | 0 (0 %) | 1198.5 | **Missing precheck** — fire-and-forget |
| `task-1776347644611-l9dqt4` | telegram_swarm | cron `0 4 * * 0` (Sundays 04:00) | NONE | 2 | 0 (0 %) | 917.6 | **Missing precheck** — too few fires for empirical confidence |
| `task-1777416775104-m5a6ls` | telegram_dedy-bukhtyat | cron `0 11 * * 5` (Fridays 11:00) | NONE | 0 | n/a | n/a | **Missing precheck** — recently added, no fires yet |
| `task-1777418999502-ii3e3v` | telegram_dedy-bukhtyat | cron `0 9 * * 5` (Fridays 09:00) | NONE | 0 | n/a | n/a | **Missing precheck** — recently added, no fires yet |

**Headline finding:** of six recurring tasks, five have no precheck at all and the sixth has one that explicitly does not gate. Effectively zero recurring fires are currently being gated out in production.

## Per-task analysis

### Heartbeat — `tessl__heartbeat` (interval 30 min)

The only task with a precheck script. The script runs roughly 8 sub-checks (unanswered messages, system health, violations, missed tasks, path hygiene, pending response, Composio email fetch, threshold-crossing review) and accumulates `wake_reasons` from each. **Since 2026-04-20 the script unconditionally appends `composio_overdue` to `wake_reasons`** (per the script's own docstring), so `wakeAgent` is always `true`. The old 1-hour `COMPOSIO_STALE_MINUTES` gate was removed for email-latency reasons.

Empirical verification: 2026-04-17 had 67/70 heartbeat fires under 10 seconds (precheck-skip duration signature). 2026-04-19 onward shows 0 sub-10s fires almost every day. The change landed and stuck.

**The trade-off** is real: keeping the precheck gateable on the OLD conditions means a Composio email outage stays invisible for up to an hour after it starts, which is the failure the script's own docstring calls out (see "Apr 16-18" reference in the source). Restoring the gate verbatim would regress on that.

**Recommendation:** revisit the heartbeat's monolithic shape rather than the gate threshold. Two architectural alternatives, in order of preference:

1. **Split Composio fetch out**: a separate lightweight scheduled task (interval 30 min) that fetches Composio email events and writes them to a state file. The heartbeat's precheck then gates on that state file plus the OLD conditions, and the dedicated fetch task itself can have a precheck that gates on "did the prior fetch succeed AND was it within the staleness window." This restores email latency AND gates the heartbeat.
2. **State-file write on gate-out**: keep one precheck script but have it write the freshly-fetched Composio events to a state file (e.g. `/workspace/state/composio-events.json`) and emit `wakeAgent: false` whenever neither the OLD conditions nor any new Composio events warrant agent attention. The next fire that DOES wake (because conditions changed or new events arrived) reads the state file as part of its first turn. This keeps email latency low (every 30 min the precheck still fetches Composio) and gates the agent on the OLD conditions plus a new "are there events worth surfacing?" check. Requires the precheck to do the Composio fetch inline — script-delegation rule allows this since the fetch is a deterministic API call, not LLM reasoning.

Either is a follow-up issue, not an in-this-PR change.

### Morning-brief — `tessl__morning-brief` (cron 07:00 daily)

13 fires in 14 days, average duration 711 s (~12 min), minimum 70 s. The 70s minimum suggests a "barely anything happened overnight" path exists; on those days the agent woke up to find an essentially-empty brief but still paid full LLM cost.

**Recommended gate inputs** (deterministic, single-script):

- Calendar events for the day-ahead window (count of confirmed events from `data/state/<group>/calendar_snapshots`, schema state-008 / #300).
- Unread emails since the last morning-brief fire (count, from email-state file or the email_seen_ids table).
- Pending undated tasks count (`pending_undated_tasks`, schema state-007 / #299).
- New cleanup items surfaced overnight (`pending_cleanup_items`).
- New decisions awaiting input (`pending_decisions`).

Wake when **any** count is non-zero. A "completely empty" morning (no events, no email, no pending items) gates the agent out. Realistic cost win: 1-3 fires per 14d on a quiet week.

### Nightly-housekeeping — `tessl__nightly-housekeeping` (cron 03:00 daily)

13 fires in 14 days, average 1198 s (~20 min), minimum 442 s (~7 min). The 442s floor is high — there's no "near-empty" run, suggesting nightly always finds work. **Likely no meaningful gate available** without changing what the skill does. Defer until the skill's responsibilities are themselves audited (out of scope for #338).

### Weekly-housekeeping — `tessl__weekly-housekeeping` (cron Sundays 04:00)

Only 2 fires in 14 days — too few for empirical confidence on the gate-out potential. Defer audit until a 30-day window is available; revisit then.

### Dedy-bukhtyat tasks (cron Fridays)

Both tasks recently created (no fires in the audit window). Defer until empirical data exists. Note for future audit: both are weekly podcast prep tasks; the gate condition is likely "is there a recording session this week" — which is exactly the kind of single-flag check that suits a precheck.

## Recommended follow-ups (will become issues)

1. **Heartbeat split** — separate Composio fetch from the gateable conditions so email latency and precheck gating both win. Architectural.
2. **Morning-brief precheck** — write `morning-brief-precheck.py` gating on calendar / email / pending-undated / cleanup / decisions counts. Concrete.
3. **Dedy-bukhtyat prechecks** — gate Friday tasks on "is there a session this week" once a few fires accumulate to confirm the gate signal. Deferred.
4. **30-day re-audit cadence** — schedule this audit to re-run quarterly so newly-added recurring tasks don't slip in without prechecks. Process / scheduled.
5. **Nightly-housekeeping responsibility audit** — separate from gating; ask whether the skill's scope is the right shape before adding a precheck. Deferred / scoped to a different epic.

The first two are the immediate cost-reduction wins; the rest are hygiene / deferred.

## Out of scope

- **Adding new recurring tasks** — this is an audit of existing ones (per the issue body).
- **A generic precheck framework** — per-task logic is the point; generic is what makes prechecks too lenient.
- **Tightening the in-container short-circuit gate** — `container/agent-runner/src/index.ts:2837-2846` already correctly short-circuits on `wakeAgent === false`. The bug isn't there; it's in the per-task scripts (or their absence).
