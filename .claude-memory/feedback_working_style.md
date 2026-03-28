---
name: Working style feedback
description: Corrections and validated approaches from Baruch across sessions
type: feedback
---

- Don't make tiles public accidentally. Set private: true from the start. Once public in Tessl, can't revert.
  **Why:** Published public versions by mistake, had to unpublish all versions and republish as private.
  **How to apply:** Always verify tile.json has "private": true before publishing.

- Don't defer/skip tasks. When asked to do something, do it — don't say "we can do this later."
  **Why:** User explicitly pushed back on deferring state consolidation, log rotation, and external heartbeat.
  **How to apply:** Execute all items in a list, don't triage them into "later."

- Commit and push frequently. User expects every change committed and pushed immediately.
  **Why:** Multiple times caught uncommitted changes.
  **How to apply:** After every logical change: build → test → commit → push → deploy to NAS.

- Deploy to NAS is part of the workflow, not a separate step. Every push should be followed by deploy.
  **How to apply:** `git push && ssh NAS "cd ~/nanoclaw && git pull && docker compose up -d --build"`

- When VPN blocks GitHub push, use `git format-patch | ssh NAS git am` to deploy directly.

- The NAS uses sudo for crontab. `crontab -e` doesn't work over SSH (terminal issue). Use pipe: `echo '...' | sudo crontab -u jbaruch -`

- Blog notes go to groups/telegram_swarm/blog-notes.md (gitignored, lives on NAS). Focus on architectural patterns and insights, not implementation details.
