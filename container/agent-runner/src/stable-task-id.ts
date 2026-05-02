/**
 * #440 — caller-supplied `task_id` format for the `schedule_task`
 * MCP tool.
 *
 * Lowercase alphanumeric + hyphens only; must start AND end with a
 * letter or digit (no leading or trailing hyphen — trailing hyphens
 * are visually indistinguishable from truncated ids in log lines);
 * hyphens permitted only in the interior; 1–64 chars total. Pinned
 * tight enough that grep / log-line splitting / shell-quoting never
 * have to worry, loose enough to encode the kinds of intent-named ids
 * long-lived recurring rows actually want
 * (`task-subskill-memory-rotation`, `task-overlay-cron-foo`,
 * `task-audit-replay-2026-q3`).
 *
 * Lives in its own module — separate from `ipc-mcp-stdio.ts` — so the
 * matching test can import it without transitively pulling in the
 * `@modelcontextprotocol/sdk` runtime that the bridge depends on (the
 * SDK only ships in `container/agent-runner/node_modules`, not at the
 * repo-level `node_modules` where vitest runs from).
 */
export const STABLE_TASK_ID_REGEX = /^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$/;
