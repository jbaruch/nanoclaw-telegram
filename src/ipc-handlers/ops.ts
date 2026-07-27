import { registerOpsFetchIpcHandlers } from './ops-fetch.js';
import { registerOpsGitIpcHandlers } from './ops-git.js';
import { registerOpsTilesIpcHandlers } from './ops-tiles.js';
import { registerOpsTzIpcHandlers } from './ops-tz.js';

/**
 * Named host operations (#845 slice 6): the ops surface — owner-tz
 * persistence, markdown fetching via the snitchmd sibling container, the
 * github backup + persona-persist git pipelines, sidecar runs, the tile
 * promote/fixup flows, and registry maintenance (tessl_update,
 * list_installed_tiles). Bodies are verbatim transplants from the legacy
 * `processTaskIpc` switch; each keeps its original in-handler
 * authorization gate (isMain re-check on the directory-verified
 * identity) and result-envelope conventions.
 *
 * #845 moved the door, not the furniture: all nine handlers landed here
 * as one ~1.2k-line blob. #879 split them by command family, leaving
 * this file as the ops composition root — the four modules below own
 * one family each, and `ipc-handlers/index.ts` still sees a single
 * `registerOpsIpcHandlers()`. The registry is a name-keyed map, so the
 * grouping changes no dispatch behavior.
 */
export function registerOpsIpcHandlers(): void {
  registerOpsTzIpcHandlers();
  registerOpsFetchIpcHandlers();
  registerOpsGitIpcHandlers();
  registerOpsTilesIpcHandlers();
}
