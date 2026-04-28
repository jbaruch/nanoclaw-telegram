---
name: session-reentry
description: Manually re-load the kill-auto-compaction checkpoint after a threshold-nuke. Use when the user asks "what was I doing before the reset" or "pick up from the checkpoint", or when starting a new session and noticing the auto-context did not include the CHECKPOINT block. The default path is auto-injected by the SessionStart hook (see `container/agent-runner/src/session-start-context.ts`); invoke this skill only when the auto-injection failed or you need to re-read the file mid-session.
---

# Session Reentry

Process steps in order; do not skip ahead.

This skill consumes `/workspace/group/.checkpoints/default.md` — the kill-auto-compaction checkpoint written by the orchestrator's threshold-nuke handler (issue #104, design at `docs/proposals/kill-auto-compaction.md`). The checkpoint's `## Facts` section enumerates state-mutating tool calls from the just-nuked session that **must NOT be re-fired**, plus pending replies and the trigger-time context summary. The optional `## Reasoning` section, if present, captures the agent's mid-task narrative before the nuke.

The default path is automatic: the SessionStart hook injects the file as a `<auto-context section="CHECKPOINT">` block on every fresh spawn, so the agent already has it in context on turn 1. This skill is the manual fallback for cases where that injection didn't fire or the file changed mid-session.

## Step 1 — Check whether a checkpoint exists

Run `scripts/check-checkpoint.sh` (relative to the skill directory). The script outputs a single JSON line with two fields:

- `exists`: `true` if `/workspace/group/.checkpoints/default.md` is present, `false` if not
- `path`: the checked path (echoed for diagnostics)

Contract: exit 0 in both cases — absence is a normal state (first-ever spawn, no recent threshold-cross, operator-cleared), not a failure. Non-zero exit means a genuine I/O fault and the skill must abort.

If `exists` is `false`, the agent was either started fresh (no recent threshold-nuke) or the operator deleted the file. **Finish here** — there is nothing to reload, and treating staleness as freshness is exactly the failure mode the design forbids (§7 of the design doc).

If `exists` is `true`, proceed immediately to Step 2.

## Step 2 — Read the checkpoint file

Use the `Read` tool on `/workspace/group/.checkpoints/default.md`. The file is plain Markdown with these expected sections:

- `# Session Checkpoint` — top-level header
- `## Facts` — orchestrator-authored, deterministic
- `### Pending replies` — message IDs awaiting reply
- `### Do NOT re-execute` — list of state-mutating tool calls from the just-nuked session
- `## Reasoning` — agent-authored narrative (optional; absent in the documented "Partial" degraded mode)

If the file is present but missing the `## Facts` heading, treat it as corrupt and **finish silently** — load nothing rather than load partial state pretending to be fresh.

If the file parsed cleanly with the `## Facts` heading present, proceed immediately to Step 3.

## Step 3 — Apply the do-NOT-re-execute list

For every entry under `### Do NOT re-execute`, treat the call as already completed. Specifically:

- A `mcp__nanoclaw__send_message` entry means the user already received that response — do not resend.
- A `mcp__nanoclaw__schedule_task` entry means the task is already on the schedule — do not re-schedule.
- A `Write` / `Edit` / `MultiEdit` entry means the file is already in its post-edit state — re-read with `Read` if you need to verify, but do not re-apply the edit.
- A `Skill` entry means the skill has already run — do not re-invoke unless the user explicitly asks for a re-run.

Read-only tool calls (`Read`, `Grep`, `Glob`, etc.) are deliberately omitted from the list: re-running them is harmless and often the right move to refresh context.

Once the do-NOT-re-execute list has been internalised (no tool call needed — this step is reasoning-only), proceed immediately to Step 4.

## Step 4 — Pick up the pending work

Use the `## Reasoning` section (when present) to identify the active task and where it was blocked. If `## Reasoning` is absent (Partial degraded mode), the agent must reconstruct from `## Facts` and any user-visible cues (the latest inbound message, the pending-replies list).

Continue from the most-actionable item. If the latest pending reply has a clear ask, address that first; otherwise resume whatever the `## Reasoning` narrative left off on.

Proceed immediately to Step 5.

## Step 5 — Confirm reentry to the user (only if asked)

If the user explicitly asked "what were we doing" or similar, summarise what the checkpoint said in 1–2 sentences and confirm what you'll do next. **If the user did NOT ask**, proceed silently — re-announcing reentry on every spawn would surface internal plumbing as if it were content. Finish here.
