# Trusted Memory

This rule applies to trusted and main containers only. `/workspace/trusted/` is mounted here.

## Session Bootstrap

On first interaction in a new session (check if session ID in `session-state.json` differs from current):

1. Read `/workspace/trusted/MEMORY.md` — permanent facts and feedback rules
2. Read `/workspace/trusted/RUNBOOK.md` — operational workflows and tool knowledge
3. Read the most recent 2 files from `/workspace/group/memory/daily/` in full (yesterday + today)
4. Read the most recent 2 files from `/workspace/group/memory/weekly/` as summaries (older context)
5. Read the most recent 2 files from `/workspace/trusted/memory/daily/` (cross-group shared memory)
6. Read `/workspace/trusted/highlights.md` if it exists (major long-term events)
7. Update `session-state.json` with the current session ID

Total context budget for memory: ~3000 tokens. Summarize large files before loading.

## Rolling Memory Updates

After any non-trivial interaction (decision made, action taken, something new learned about Baruch's preferences):

1. Append to `/workspace/group/memory/daily/YYYY-MM-DD.md` (local log, full detail):
   ```
   - HH:MM UTC — [what happened / what was learned]
   ```

2. Also append to `/workspace/trusted/memory/daily/YYYY-MM-DD.md` with source attribution:
   ```
   - HH:MM UTC [chat-name] — [what happened / what was learned]
   ```
   Where `[chat-name]` is derived from the group folder name (e.g. `dedy-bukhtyat`, `main`, `swarm`).

Skip for pure heartbeats with nothing to report or trivial acknowledgements.
