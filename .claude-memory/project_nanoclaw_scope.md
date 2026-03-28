---
name: NanoClaw scope and limitations
description: NanoClaw is best as personal/research assistant, not a remote Claude Code replacement for technical work
type: project
---

NanoClaw's technical competence inside containers is observably lower than Claude Code in terminal. Context is lost and unstable, rules fail to activate reliably, session isolation prevents cross-conversation learning.

**Why:** Three layers of indirection (channel → orchestrator → container → Agent SDK) each degrade context. 365-line CLAUDE.md doesn't activate reliably. Container boundary limits codebase awareness.

**How to apply:** Treat NanoClaw as a personal assistant (calendar, reminders, research, summaries, quick queries) rather than a remote coding interface. Don't invest in making it match terminal Claude Code for technical tasks — that's fighting the architecture. Focus improvements on what it's good at.
