---
name: soul-review
description: Weekly self-improvement review for SOUL.md. Reads recent memory and feedback files, identifies new patterns about Baruch's preferences and communication style, and proposes targeted updates to SOUL.md for his review. Use when running the scheduled Monday workflow, or when manually triggered with phrases like "review soul", "soul update", "self-improvement review", "update my preferences", or "review my profile". Runs after nightly-housekeeping.
---

# Soul Review Skill

Runs weekly to propose updates to `/workspace/global/SOUL.md` based on what was learned from recent interactions.

## When to run
Monday nights, after nightly-housekeeping completes. Can also be triggered manually: "review soul", "soul update", "self-improvement review".

## Step 1: Load context

Read the following in parallel:
- `/workspace/global/SOUL.md` — current identity and personality definition
- Last 2 files from `/workspace/group/memory/weekly/` — recent weekly summaries
- All files from `/home/node/.claude/projects/-workspace-group/memory/` matching `feedback_*.md` — accumulated feedback
- `/workspace/group/memory/daily/` — last 3 days of daily logs

## Step 2: Analyze for patterns

Look for evidence that SOUL.md is incomplete, wrong, or outdated:

- **New preferences discovered** — things Baruch consistently likes or dislikes that aren't captured
- **Communication patterns** — phrases, topics, or tone he uses that aren't in the voice profile
- **Noise rules learned** — new categories of emails/events that should be silently skipped
- **People** — new contacts, relationship context not yet in key-people.md
- **Projects** — new active projects, completed projects to remove
- **Corrections** — places where current SOUL.md led to wrong behavior (based on feedback)
- **What NOT to do** — new anti-patterns identified from his corrections

## Step 3: Draft proposals

For each meaningful insight, draft a concrete proposed change:

```
PROPOSED CHANGE #N
Section: [which section of SOUL.md]
Current: "[exact current text, or MISSING if new]"
Proposed: "[exact replacement or addition]"
Reason: [1 sentence why — what behavior/feedback drove this]
```

Only propose changes that are:
- Supported by at least 2 data points (not a single incident)
- Concrete and actionable (not vague)
- Actually absent from current SOUL.md

Skip anything already covered. Skip trivial rewording.

## Step 4: Send for review

If there are proposals, send via `mcp__nanoclaw__send_message` (no reply_to — proactive):

```
🧠 <b>Weekly soul review</b> — N proposed updates

[PROPOSED CHANGE #1]
...

Reply with the number(s) to apply, or ignore to skip all.
```

If there are no meaningful proposals → output nothing. Silence means soul.md is already accurate.

## Step 5: Apply approved changes

When Baruch responds with approval (e.g. "apply 1, 3" or "all"):
- Edit `/workspace/global/SOUL.md` directly with the approved changes
- Confirm: "Applied #1, #3."
- Append to daily memory log: `- HH:MM UTC — soul-review: applied N updates to SOUL.md`
