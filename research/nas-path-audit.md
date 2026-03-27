# NAS Path Audit
_2026-03-27 — paths that break when /workspace/project mount is removed_

## Current: Agent reads from /workspace/project (read-only mount of NanoClaw source)

On NAS, this mount goes away. Agent only has `/workspace/group/` (read-write) and `/workspace/ipc/` (read-write).

## Affected Resources

### messages.db (`/workspace/project/store/messages.db`)
**Used by:** check-system-health (3 SQL queries), check-unanswered (1 SQL query), manage-groups (1 SQL query)
**Resolution:** Mount `store/` read-only at `/workspace/store/` in agent containers. One new mount, minimal surface area. Or move SQL checks to external heartbeat.

### logs (`/workspace/project/logs/`)
**Used by:** check-system-health (log size, error counting, retry exhaustion)
**Resolution:** Move log checks to external heartbeat entirely. The agent doesn't need to read host logs — that's what the external heartbeat is for.

### sessions (`/workspace/project/data/sessions/`)
**Used by:** check-system-health (session bloat, session cleanup)
**Resolution:** Move session checks to external heartbeat. Session directories are host-side infrastructure.

### IPC (`/workspace/project/data/ipc/`)
**Used by:** check-system-health (stuck _close files)
**Resolution:** Agent already has `/workspace/ipc/` mounted. Update path from `/workspace/project/data/ipc/` to `/workspace/ipc/`.

### registered_groups.json (`/workspace/project/data/registered_groups.json`)
**Used by:** manage-groups (read/write for group registration)
**Resolution:** Groups are managed via MCP tools (register_group, list_tasks). The agent doesn't need direct file access — the MCP server handles it. Remove file-path references from manage-groups skill.

### SOUL.md (`/workspace/project/groups/global/SOUL.md`)
**Used by:** core-behavior rule (identity)
**Resolution:** Already handled — rule checks `/workspace/global/SOUL.md` first (non-main mount), falls back to project path. On NAS, SOUL.md will be baked into the image at `/workspace/global/SOUL.md`.

### Global CLAUDE.md (`/workspace/project/groups/global/CLAUDE.md`)
**Used by:** core-behavior rule (global memory)
**Resolution:** Global memory becomes part of the image or a mounted volume. The "update global memory" instruction may need rethinking on NAS.

### HEARTBEAT.md (`/workspace/project/groups/global/HEARTBEAT.md`)
**Used by:** was the old heartbeat prompt, now replaced by tile skill. No longer referenced.
**Resolution:** Already resolved — heartbeat is a tile skill now.

## Migration Strategy

### Phase 1: Move infrastructure checks to external heartbeat (no agent changes)
- DB size, staleness, growth → already in external heartbeat
- Log size, errors → already in external heartbeat
- Session bloat → add to external heartbeat
- Stuck IPC files → add to external heartbeat
- Orphaned containers → already in external heartbeat

### Phase 2: Simplify agent check-system-health
After Phase 1, the agent's check-system-health only needs:
- Stuck scheduled tasks → needs DB access OR move to external heartbeat
- OneCLI health → curl to host gateway (works from container)
- Retry exhaustion → needs log access OR move to external heartbeat

### Phase 3: Add store mount for remaining DB queries
If check-unanswered and manage-groups still need direct DB access:
- Mount `store/` read-only at `/workspace/store/`
- Update paths in skills
- This becomes the 4th mount (group, ipc, sessions, store)

### Alternative: Move ALL infrastructure checks to external heartbeat
The agent's heartbeat becomes: invoke /check-unanswered, /check-calendar, /check-email (all use MCP/Composio, not filesystem). System health checks are entirely external. Cleanest separation.
