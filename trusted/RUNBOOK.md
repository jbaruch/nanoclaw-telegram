# AyeAye Operational Runbook

How things work. Updated by nightly-housekeeping from daily_discoveries.

## GitHub — Blog Notes
- **File:** `blog-notes.md` in root of `jbaruch/nanoclaw`, **branch: main**
- **How to update:** Composio GitHub `GITHUB_CREATE_OR_UPDATE_FILE_CONTENTS` (owner: jbaruch, repo: nanoclaw, branch: main)
- **Permission:** Push without approval ("ты туда можешь фигачить даже без моего апрувала")
- **NOT** the backup branch (backup branch is for agent state backup only)

## GitHub — Backup Repo
- **What:** Agent state backup pushed to `backup` branch of `jbaruch/nanoclaw`
- **How:** `mcp__nanoclaw__github_backup` tool (host-side)
- **Does NOT update main branch** — separate from blog-notes workflow

## Memory Architecture
- `/workspace/trusted/MEMORY.md` — index of persistent facts and feedback rules
- `/workspace/trusted/RUNBOOK.md` — this file, operational workflows
- `/workspace/trusted/memory/daily_discoveries.md` — immediate captures, triaged nightly
- `/workspace/trusted/memory/daily/YYYY-MM-DD.md` — narrative daily logs
- `/workspace/trusted/memory/weekly/` — weekly summaries

## Session Bootstrap Order
1. MEMORY.md (always)
2. RUNBOOK.md (always)
3. Last 2 daily logs
4. Last 2 weekly summaries
5. trusted/highlights.md if exists

## Email — J.P. Morgan
- "New online notice" emails = mandatory legal docs (ADV2B etc.) to ~15 brokerage accounts. Do NOT report. Only flag: construction draws, payment failures, fraud.

## Email — amir@sadogursky.com
- Family mailing group (Baruch + Alice). Amazon/promo → ignore. School tuition, school notices, time-sensitive family items → flag.

## Composio Session IDs
- Pass `session_id` in ALL subsequent meta tool calls after COMPOSIO_SEARCH_TOOLS
- Re-use session within a workflow, generate new for new workflow
