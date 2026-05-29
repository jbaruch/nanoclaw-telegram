/**
 * #645 — byAir MCP server registration for on-demand flight lookup.
 *
 * byAir is a personal MCP endpoint (`api.byairapp.com/mcp`) whose API
 * key is carried inline in the URL query string (`BYAIR_MCP_URL`). The
 * precheck loop in the `jbaruch/nanoclaw-flight-assist` tile reaches it
 * via the Python `ByAirClient` HTTP wrapper and filters the ~13KB raw
 * response down to a ~1KB operational slice before any state write — so
 * the polling path never put byAir in front of the agent. This helper
 * registers byAir as a Claude MCP tool so the agent can answer ad-hoc
 * questions ("has Amir landed?") on demand.
 *
 * Raw, not a filtering shim: the use is occasional and pull-based, so
 * the per-call ~13KB context cost is acceptable (the shim's ~1KB slice
 * only pays off under a frequent loop, which this is not).
 *
 * The registration is deliberately NOT `alwaysLoad` — byAir's tool
 * surface stays deferred behind a ToolSearch hop, keeping it out of the
 * always-on context and off the proactive precheck/wake loop.
 *
 * Pure module — separate from `index.ts` — so the test imports it
 * without pulling in the Agent SDK runtime. Same pattern as
 * `overlay-tiles.ts`.
 */

export interface ByairMcpServer {
  type: 'http';
  url: string;
}

/**
 * Build the byAir entry for the agent's `mcpServers` config. Returns a
 * single-key `{ byair: { type: 'http', url } }` map when `byairUrl` is a
 * non-empty string, else `{}` so the spread is a no-op where the URL is
 * absent (untrusted tiers, or chats without the flight-assist overlay).
 *
 * No auth header: the API key lives in the URL query string, mirroring
 * how the precheck consumes `BYAIR_MCP_URL`.
 */
export function byairMcpServer(
  byairUrl: string | undefined,
): Record<string, ByairMcpServer> {
  if (!byairUrl) return {};
  return { byair: { type: 'http', url: byairUrl } };
}
