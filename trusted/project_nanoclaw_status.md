---
name: NanoClaw current project status
description: Active feature status, what works, what's broken, what to do next
type: project
---

External heartbeat running on NAS host at `/home/jbaruch/nanoclaw/scripts/heartbeat-external.sh`, cron every 15min. Logrotate hourly.

**Why:** Infrastructure moved to NAS, host-level scripts handle what containers can't.
**How to apply:** check-system-health and check-unanswered have external triggers now.

Voice transcription: working.
agent-browser: installed, not proactively used — remember to use it for web research tasks.
Calendar: read-write (we created a Voxxed Amsterdam event).
Gmail: read-write (will reply to emails).

Research docs on disk:
- /workspace/group/swarm-research.md — 5 swarm use cases (Talk Gauntlet, Research Blitz, Draft Assassins, Gauntlet Q&A, Spec Scrutiny)
- /workspace/group/unused-features-research.md — unused features audit

Pending:
- check-unanswered: verify external heartbeat calls it correctly
- Composio integrations idle: Drive, YouTube, Notion, Discord
- LinkedIn @mentions: будет через webhooks (https://learn.microsoft.com/en-us/linkedin/shared/api-guide/webhook-validation) — отложено на потом
- nightly-housekeeping at 3am: sync-tripit running
