# NanoClaw Admin Context

These rules apply only to the main channel, which has elevated privileges.

## Elevated Privileges

This is the **main channel**. You can manage groups, schedule tasks for other groups, and access the full project filesystem (read-only).

## Authentication

Anthropic credentials must be either an API key from console.anthropic.com (`ANTHROPIC_API_KEY`) or a long-lived OAuth token from `claude setup-token` (`CLAUDE_CODE_OAUTH_TOKEN`). Short-lived tokens from the system keychain or `~/.claude/.credentials.json` expire within hours and can cause recurring container 401s. The `/setup` skill walks through this. OneCLI manages credentials (including Anthropic auth) — run `onecli --help`.

## Container Mounts

Main has read-only access to the project and read-write access to its group folder:

| Container Path | Host Path | Access |
|----------------|-----------|--------|
| `/workspace/project` | Project root | read-only |
| `/workspace/group` | `groups/{folder}/` | read-write |

Key paths inside the container:
- `/workspace/project/store/messages.db` — SQLite database (messages, chats, registered_groups)
- `/workspace/project/groups/` — All group folders

## Scheduling for Other Groups

When scheduling tasks for other groups, use the `target_group_jid` parameter with the group's JID from `registered_groups`:
- `schedule_task(prompt: "...", schedule_type: "cron", schedule_value: "0 9 * * 1", target_group_jid: "120363336345536173@g.us")`

The task will run in that group's context with access to their files and memory.
