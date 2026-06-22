/**
 * #305 follow-up — pure helpers for the per-chat tile-overlay MCP
 * surface.
 *
 * Lives in its own module — separate from `ipc-mcp-stdio.ts` — so the
 * matching test can import these helpers without transitively pulling
 * in the `@modelcontextprotocol/sdk` runtime that the bridge depends
 * on (the SDK only ships in `container/agent-runner/node_modules`,
 * not at the repo-level `node_modules` where vitest runs from). Same
 * pattern as `stable-task-id.ts` (#440).
 */

/**
 * Args accepted by the `register_group` MCP tool that contribute to
 * `containerConfig`. Mirrors the Zod schema in `ipc-mcp-stdio.ts` —
 * keep the two in sync.
 */
export interface RegisterGroupContainerArgs {
  trusted?: boolean;
  enableHeartbeat?: boolean;
  additionalMounts?: Array<{
    hostPath: string;
    containerPath?: string;
    readonly?: boolean;
  }>;
  additionalTiles?: string[];
}

/**
 * Build the `containerConfig` payload field for a `register_group`
 * IPC request. Returns `undefined` when no contributing field is set,
 * so the resulting IPC payload's `containerConfig` slot is absent
 * rather than `{}` — the host treats undefined as "no per-group
 * config" and `{}` as "explicit empty config", which differ on the
 * untrusted-by-default semantics.
 *
 * `additionalTiles` is gated on length so an empty array doesn't
 * spuriously create a `containerConfig` object — empty array means
 * "no overlay" which is the same as "field absent" at this layer.
 */
export function buildRegisterGroupContainerConfig(
  args: RegisterGroupContainerArgs,
): Record<string, unknown> | undefined {
  const hasTrusted = args.trusted !== undefined;
  const hasHeartbeat = args.enableHeartbeat !== undefined;
  const hasMounts = !!args.additionalMounts;
  const hasOverlay = !!(
    args.additionalTiles && args.additionalTiles.length > 0
  );

  if (!hasTrusted && !hasHeartbeat && !hasMounts && !hasOverlay) {
    return undefined;
  }

  return {
    ...(hasTrusted ? { trusted: args.trusted } : {}),
    ...(hasHeartbeat ? { enableHeartbeat: args.enableHeartbeat } : {}),
    ...(hasMounts ? { additionalMounts: args.additionalMounts } : {}),
    ...(hasOverlay ? { additionalTiles: args.additionalTiles } : {}),
  };
}

/**
 * Args accepted by the `set_additional_tiles` MCP tool. Mirrors the
 * Zod schema in `ipc-mcp-stdio.ts`.
 */
export interface SetAdditionalTilesArgs {
  groupFolder: string;
  additionalTiles: string[] | null;
}

/**
 * Render the human-readable `desc` segment for the
 * `set_additional_tiles` MCP-tool response. `null` and `[]` both
 * surface as "cleared (baseline only)" — the host treats them
 * identically and the operator-facing message should match.
 */
export function describeOverlayUpdate(
  additionalTiles: string[] | null,
): string {
  if (additionalTiles === null || additionalTiles.length === 0) {
    return 'cleared (baseline only)';
  }
  return `[${additionalTiles.join(', ')}]`;
}
