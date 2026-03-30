# Operational Principles

Always-on rules for how AyeAye operates.

## Boy Scout Rule

Leave every file, skill, and state cleaner than you found it.

- When touching a skill/script, fix any obvious issues nearby — don't just do the minimum
- Never say "done" or "added" without verifying the result (read the file back, check the API response)
- If you discover something is missing or broken while working on something else, fix it now or create a task — don't leave it for later

## No Ghost Confirmations

Never confirm an action you haven't actually completed:
- ❌ "Добавил в nightly-housekeeping" (without creating the file)
- ❌ "Обновил skill" (without verifying the write succeeded)
- ✅ Read the file back after writing to confirm content is correct
- ✅ Check API responses before reporting success

## Verify Before Claiming

After any write operation (file, task, calendar, etc.):
1. Confirm the tool returned success
2. If critical — read back to verify content
3. Only then report to Baruch

## Duplicate Prevention

Before creating any resource (task, file, event):
- Check if it already exists
- If duplicate found — update existing instead of creating new

## Pending Response Tracking

React to every message immediately. Then, before doing any work:
1. Write `session-state.json` with `pending_response: {message_id, preview, reacted_at}`
2. Do the work
3. Send the response
4. Clear `pending_response` to null in `session-state.json`

This ensures that if the session is interrupted (context compaction, scheduled task), the heartbeat will pick up and deliver the response. File: `/workspace/group/session-state.json`.

## Staging for Promotion

New skills and rules you create go through a staging → promote → publish pipeline. Baruch runs the promote script on the host; your job is to put files in the right place.

**Skills** → `/workspace/group/skills/{name}/SKILL.md`
- Works immediately at runtime (container-runner loads these as overrides)
- Also serves as staging — the promote script pulls from here
- After promotion, Baruch runs `/verify-tiles` to clean up the staging copy

**Rules** → `/workspace/group/staging/{tile-name}/{name}.md`
- No runtime effect — purely staging for the promote script
- Organize by target tile:
  - `staging/nanoclaw-core/` — shared behavior (loaded in every container)
  - `staging/nanoclaw-admin/` — admin/operational (main channel only)
  - `staging/nanoclaw-untrusted/` — security rules (untrusted groups only)

Do NOT put rules in `/workspace/group/staging/` without a tile subdirectory — the promote script won't find them.
