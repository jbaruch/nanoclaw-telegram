---
name: check-watchlist
description: Checks tracked upcoming TV shows in watchlist.json and sends a Telegram notification message via MCP when any have been released. Use when running nightly release checks, monitoring streaming release dates, or checking whether new episodes or shows from a watchlist are now available to watch. Called nightly from nightly-housekeeping.
---

# Check Watchlist Skill

Monitors `/workspace/group/watchlist.json` for upcoming shows and notifies when they release.

## Step 1: Read watchlist

Read `/workspace/group/watchlist.json`. If the file doesn't exist or `tracking` array is empty, exit silently.

Filter to shows where `notified: false` only.

## Step 2: Check release status

For each unnotified show, do a web search:
```
"[title]" release date 2025 2026 streaming
```

Determine:
- **Released**: show has premiered on its platform and episodes are available
- **Not yet released**: still in production or announced without air date
- **Cancelled**: show was cancelled before release

## Step 3: Handle results

**If released:**
1. Compose a short notification message (Telegram HTML format):
   ```
   📺 <b>[Title]</b> is now available on [Platform]!
   [1 sentence why Baruch will like it, from the `reason` field]
   ```
2. Send via `mcp__nanoclaw__send_message` (standalone, not a reply — this is a proactive alert)
3. Update watchlist.json: set `notified: true` and add `"released": "YYYY-MM-DD"` (actual release date if known, today's date otherwise)
4. Write the updated watchlist.json back to disk

**If not yet released:** Stay completely silent. Do not update the file.

**If cancelled:**
1. Update watchlist.json: set `notified: true` and add `"cancelled": true`
2. Do NOT notify Baruch — a cancelled show is not actionable

## Step 4: Write back

After processing all shows, if any were updated, write the modified watchlist.json back.
Read the full file first, update only the changed entries, write the complete file back.

## Notes
- This skill runs nightly — it should be fast (one search per unnotified show)
- Only notify for actual releases, not renewal announcements or trailers
- If the show has new season announced but no air date confirmed, treat as "not yet released"
- Silence is the default — only speak when a show is actually available to watch
