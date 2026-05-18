# Maintenance cost audit — 2026-05-14

Refresh of the analysis in [#509](https://github.com/jbaruch/nanoclaw/issues/509). The original issue body (2026-05-05) baselined maintenance at ~$288/day on Opus. That baseline is stale — maintenance was demoted to Sonnet between then and now, dropping daily spend by ~95%. This document refreshes the per-task picture against current `usage.jsonl` and revises the Phase 2/3 plan.

## Method

- Pulled `logs/usage.jsonl` rows from `nas` for the trailing 7 days (3470 rows).
- Pulled `task_run_logs` rows for the same window (757 fires).
- Attributed each maintenance usage row to a task via nearest-prior `task_run_logs` row within 10 minutes (the usage row's API call window extends past `duration_ms`, which only covers the precheck-to-success interval, so a strict `[start, start+dur]` window misses most calls — see attribution-quirk note below).

## Headline numbers, 2026-05-07 → 2026-05-14

Total: **$345.27 / 3470 calls** across all sessions.

Split by slot:

| Slot | Model | Calls | USD | $/call |
|---|---|---:|---:|---:|
| `default` (`<inbound>`) | Opus | 1102 | $170.60 | $0.155 |
| `maintenance` | Sonnet (96%) | 1623 | $170.55 | $0.105 |
| `gates/haiku-classifier` | Haiku | 665 | $1.65 | $0.0025 |

Maintenance and inbound are roughly equal — the $288/day-on-Opus baseline in #509 is no longer the picture.

## Per-task maintenance breakdown

| Task | Calls | $/7d | $/call | $/day | Notes |
|---|---:|---:|---:|---:|---|
| `tessl__composio-fetch` | 675 | $104.24 | $0.154 | $14.89 | `*/30 * * * *`, 61% precheck-gated |
| `tessl__heartbeat` | 695 | $52.81 | $0.076 | $7.54 | `*/30 * * * *`, 27% precheck-gated |
| `tessl__morning-brief` | 43 | $2.23 | $0.052 | $0.32 | daily |
| `task-blocklist-watchdog-pr533` | 8 | $0.95 | $0.119 | $0.14 | daily |
| All others | <30 ea. | <$0.50 ea. | — | <$0.10 ea. | |
| **Maintenance total** | **1623** | **$170.55** | — | **$24.36** | |

**87% of maintenance spend is composio-fetch + heartbeat.** Every other scheduled task combined is <$2/day.

## Surprising finding: composio is more expensive per call than heartbeat

composio-fetch averages $0.154/call vs heartbeat's $0.076 — double — despite being the simpler task (fetch JSON from Composio, write to disk). Two contributing factors:

- Both spawn cold every 30 min. The system-prompt + skill-load overhead is a fixed cost per spawn.
- composio's transcripts are longer because the tool output is bulky (full event payload) and gets re-read by the agent.

Demoting composio-fetch to Haiku addresses both — Haiku is ~12x cheaper per input token than Sonnet, so the cold-spawn overhead drops proportionally.

## Tier recommendations

| Task | Current | Target | Justification |
|---|---|---|---|
| `tessl__composio-fetch` | Sonnet | **Haiku** | Deterministic fetch-and-write. No reasoning. ~$14/day saving. |
| `tessl__heartbeat` | Sonnet | Sonnet (keep) | Does triage logic over recent logs. Haiku is borderline; defer until composio lands. |
| `tessl__morning-brief` | Sonnet | Sonnet (keep) | Synthesis of overnight context. |
| `tessl__nightly-external-sync` | Sonnet | Sonnet (keep) | Calendar + reclaim sync, light reasoning. Cost too low to justify migration. |
| `tessl__memory-rotation` | Sonnet | Sonnet (keep) | Synthesis. Cost too low to justify migration. |
| Other nightlies | Sonnet | Sonnet (keep) | Each <$0.50/day; not worth per-task config drift. |

**Expected post-rollout maintenance spend: ~$10/day** (heartbeat $7.54 + others ~$2), down from $24/day.

## Phase 2 is shipped; Phase 3 is the open work

#509's Phase 2 (per-group `maintenanceAgentModel` override) has already landed — see `resolveSessionAgentModel` at `src/container-runner.ts:726`. That's what dropped maintenance off Opus and onto Sonnet, accounting for the 95% baseline reduction between the issue's 2026-05-05 numbers and today's.

With the refreshed data, Phase 2's knob can't express the right next step: composio (target: Haiku) and heartbeat (target: Sonnet) share the same maintenance session in the same group. Per-session granularity is too coarse to demote composio without also demoting heartbeat.

**Implement Phase 3 (per-task `agent_model` column on `scheduled_tasks`):**

- Add `agent_model TEXT NULL` column to `scheduled_tasks` (issue body's existing Phase 3 plan).
- Spawn-time resolution (preserves Phase 2's `maintenanceAgentModel` step that landed in PR #511 — earlier drafts of this doc omitted it): per-row `agent_model` → `maintenanceAgentModel` (when `sessionName === 'maintenance'`) → group `agentModel` → `process.env.AGENT_MODEL` → `DEFAULT_AGENT_MODEL`. The new audit-log source for the top step is `task_override`. An unknown-prefix per-row value falls back to the **session-level** value (the maintenance override on a maintenance spawn, the group override otherwise), NOT to the global default — a typo on a Sonnet-pinned maintenance slot routes back to Sonnet, not back to Opus.
- Cadence-registry rows learn an optional `agentModel` field, flowing into the column at registry sync via the existing shape-change UPDATE.
- Writer contention: the `set_task_agent_model` IPC handler refuses imperative writes to cadence-registry-sourced rows (the cadence-registry rebuild's shape-change UPDATE would silently revert them on the next tile-touching spawn). Operators who want to flip a cadence-owned task modify the SKILL.md `agentModel:` frontmatter and republish the tile; the imperative IPC path is for `source = 'schedule-task'` rows only.
- First migration: `tessl__composio-fetch` → `claude-haiku-4-5-20251001` via a follow-up `jbaruch/nanoclaw-admin` PR adding `agentModel: claude-haiku-4-5-20251001` to the skill's SKILL.md frontmatter.
- Observe 7 days. If composio still produces correct output (events.json freshness, downstream heartbeat consumption), the design is validated and the same lever can be applied to other tasks if cost grows.

## #569 (Batches API) — defer

#569's premise was 50% off for async-tolerant scheduled work. With composio and heartbeat correctly classified as **not** async-tolerant (composio feeds heartbeat 30 min later; heartbeat is an alarm), the remaining batchable tasks are nightly housekeeping — combined cost <$1/day. The infrastructure for durable batch tracking + 24h polling + restart-resilience exceeds the saving at this volume.

Revisit when:

- Bulk retroactive workloads land (per #569's "Out of scope" list of historical processing).
- A scheduled-skill class emerges with both high volume and a 24h SLA.

## Inbound = the next lever

Inbound `default`-session calls on Opus: **$170/7d = $24/day**. Same magnitude as the entire maintenance bucket. Today every inbound turn runs Opus regardless of complexity. The Haiku classifier (`gates/haiku-classifier`, $0.0025/call) already fronts inbound for gate decisions, so the routing infrastructure exists. A separate issue should scope "inbound model tiering by classifier verdict" — out of scope for #509.

## Attribution-quirk note

`task_run_logs.duration_ms` measures precheck-to-success time, not full agent runtime. Usage rows for a task extend ~40s past `start + duration_ms` because the agent's API calls continue after the row is written. Attribution used a 10-min trailing window from each task's `run_at`; 185 of 1623 maintenance rows (11%) remained unattributed, mostly clustered around task transitions or container restarts. The 185-row unattributed bucket totals $9.13 (5% of maintenance), small enough that the per-task ranking is robust.

If this becomes a recurring analysis, the cleaner fix is to plumb the scheduled-task ID through `registerContainer` (currently `task_id: sessionName` at `src/container-runner.ts:3091` — just `'maintenance'`) so usage rows carry the actual task identifier. Filed as a follow-up note in the #509 update comment.
