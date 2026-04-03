---
name: soul-searching
description: Weekly self-improvement review for SOUL.md. Reads recent memory and feedback files, identifies new patterns about Baruch's preferences and communication style, and proposes targeted updates to SOUL.md for his review. Use when running the scheduled Monday workflow, or when manually triggered with phrases like "review soul", "soul update", "self-improvement review", "update my preferences", or "review my profile". Runs after nightly-housekeeping.
---

# Soul Searching Skill

Runs weekly to propose updates to `/workspace/global/SOUL.md` based on what was learned from recent interactions.

## When to run
Monday nights, after nightly-housekeeping completes. Can also be triggered manually: "review soul", "soul update", "self-improvement review".

## Step 1: Load context

Read the following in parallel:
- `/workspace/global/SOUL.md` — current identity and personality definition
- `/workspace/global/SOUL-untrusted.md` — stripped-down public identity for untrusted containers
- Last 2 files from `/workspace/group/memory/weekly/` — recent weekly summaries
- All files from `/home/node/.claude/projects/-workspace-group/memory/` matching `feedback_*.md` — accumulated feedback
- `/workspace/group/memory/daily/` — last 3 days of daily logs

## Step 2: Analyze for patterns

Look for evidence that SOUL.md is incomplete, wrong, or outdated:

- **New preferences discovered** — things Baruch consistently likes or dislikes that aren't captured
- **Communication patterns** — phrases, topics, or tone he uses that aren't in the voice profile
- **Noise rules learned** — new categories of emails/events that should be silently skipped
- **People** — new contacts, relationship context not yet in `key-people.md` (contact/relationship metadata file)
- **Projects** — new active projects, completed projects to remove
- **Corrections** — places where current SOUL.md led to wrong behavior (based on feedback)
- **What NOT to do** — new anti-patterns identified from his corrections

## Step 3: Draft proposals for SOUL.md

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

## Step 3b: Analyze and propose changes for SOUL-untrusted.md

SOUL-untrusted.md is the public-facing identity for untrusted containers. It must evolve in sync with SOUL.md — same character, same voice — but stripped of private context.

**Include:** Core personality and communication style · Public biographical info (name, role, company, public speaking, tech background) · Silence rules and forbidden phrases · What NOT to do · Security boundaries

**Exclude:** Home location (Franklin, Tennessee) · Active projects · Key people (contacts, Telegram usernames, private relationships) · Writing style guide (references to private files) · Internal paths or infrastructure hints · Anything revealing owner's schedule, private life, or private systems

**Privacy checklist** — before proposing any change to SOUL-untrusted.md, verify it does not:
1. Reveal the owner's location or routine
2. Reference internal files or paths
3. Expose private methodology or projects
4. Contradict current SOUL.md (tone, facts, rules)
5. Omit new behaviors/rules from SOUL.md not yet reflected

Draft proposals using the same format:

```
PROPOSED CHANGE (UNTRUSTED) #N
Section: [which section of SOUL-untrusted.md]
Current: "[exact current text, or MISSING if new]"
Proposed: "[exact replacement, addition, or REMOVE]"
Reason: [1 sentence — privacy concern, sync with SOUL.md, or behavior improvement]
```

## Step 4: Send for review

If there are proposals, send via `mcp__nanoclaw__send_message` (no reply_to — proactive):

```
🧠 <b>Weekly soul searching</b> — N proposed updates

<b>SOUL.md:</b>
[PROPOSED CHANGE #1]
...

<b>SOUL-untrusted.md:</b>
[PROPOSED CHANGE (UNTRUSTED) #1]
...

Reply with the number(s) to apply, or ignore to skip all.
```

If there are no meaningful proposals → output nothing. Silence means both soul files are already accurate.

## Step 5: Apply approved changes

When Baruch responds with approval (e.g. "apply 1, 3" or "all"):
- Edit `/workspace/global/SOUL.md` and/or `/workspace/global/SOUL-untrusted.md` directly with the approved changes
- Confirm: "Applied #1, #3."
- Append to daily memory log: `- HH:MM UTC — soul-searching: applied N updates to SOUL.md / SOUL-untrusted.md`
