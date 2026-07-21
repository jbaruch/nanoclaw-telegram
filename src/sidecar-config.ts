import fs from 'fs';
import path from 'path';

import { isFsErrorWithCode } from './fs-errors.js';
import type { SidecarSpec } from './sidecar-runner.js';

/**
 * Data-driven sidecar registry loader (#850, epic #844).
 *
 * Sidecar specs are host config, not source code: `sidecar-runner.ts`
 * used to hard-code personal NAS paths (the audible-backup mounts) in
 * TypeScript. They now live in a JSON file on the host —
 * `config/sidecars.json` by default, overridable via
 * `SIDECARS_CONFIG_PATH` — that is gitignored, so a deployment adds or
 * edits sidecars without a TS change. `config/sidecars.example.json`
 * documents the shape; `config/README.md` documents the workflow and
 * the security model.
 *
 * Trust model is unchanged from #750: this file is TRUSTED HOST CONFIG.
 * The image, mounts, and flag allowlist come from here, never from the
 * IPC payload — a compromised container can still only name a
 * registered sidecar and append allowlisted flags.
 *
 * `${VAR}` placeholders in MOUNT strings expand from the host
 * environment at load time, plus one built-in: `HOST_PROJECT_PARENT`,
 * the directory containing the project root (preserves the pre-#850
 * audible-backup `.audible` mount resolution). An unknown variable is a
 * config error, never a silent empty string. Image and args stay
 * literal — only mount paths are machine-specific.
 */

/** Errnos treated as "file state" when reading the config (fail loudly, typed). */
const CONFIG_FS_CODES = ['EACCES', 'EPERM', 'EISDIR', 'ENOTDIR', 'ELOOP'];

const DEFAULT_TIMEOUT_MS = 600_000;
const DEFAULT_MAX_BUFFER = 10 * 1024 * 1024;

export function sidecarsConfigPath(): string {
  // Resolved against the orchestrator's working directory (`/app` in
  // the container, where docker-compose bind-mounts `./config`
  // read-only) — NOT `HOST_PROJECT_ROOT`, which is the host-side path
  // used only for composing docker `-v` mount strings the daemon
  // evaluates on the host. This process reads the FILE through its own
  // filesystem.
  return (
    process.env.SIDECARS_CONFIG_PATH ||
    path.join(process.cwd(), 'config', 'sidecars.json')
  );
}

export type SidecarRegistryResult =
  | { ok: true; registry: Record<string, SidecarSpec>; configPath: string }
  | { ok: false; error: string; configPath: string };

/**
 * Load and validate the sidecar registry from host config.
 *
 * - File absent → `ok: true` with an EMPTY registry. A platform install
 *   with no sidecars is a normal state; `runSidecar` surfaces the
 *   config path in its unknown-name error so the fix is discoverable.
 * - File present but unreadable / not JSON / wrong shape / failed
 *   `${VAR}` expansion → `ok: false` with an actionable, entry-specific
 *   message. Never a silently-empty registry.
 */
export function loadSidecarRegistry(): SidecarRegistryResult {
  const configPath = sidecarsConfigPath();
  const fail = (error: string): SidecarRegistryResult => ({
    ok: false,
    error,
    configPath,
  });

  let rawText: string;
  try {
    rawText = fs.readFileSync(configPath, 'utf-8');
  } catch (err) {
    if (isFsErrorWithCode(err, ['ENOENT'])) {
      return { ok: true, registry: {}, configPath };
    }
    if (!isFsErrorWithCode(err, CONFIG_FS_CODES)) throw err;
    return fail(
      `cannot read ${configPath} (${(err as NodeJS.ErrnoException).code}) — fix the file permissions or point SIDECARS_CONFIG_PATH at a readable file`,
    );
  }

  let raw: unknown;
  try {
    raw = JSON.parse(rawText);
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
    return fail(
      `${configPath} is not valid JSON (${err.message}) — see config/sidecars.example.json for the expected shape`,
    );
  }

  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    return fail(
      `${configPath} must be a JSON object mapping sidecar names to specs — see config/sidecars.example.json`,
    );
  }

  const registry: Record<string, SidecarSpec> = {};
  for (const [name, entry] of Object.entries(raw as Record<string, unknown>)) {
    const where = `${configPath}: "${name}"`;
    if (entry === null || typeof entry !== 'object' || Array.isArray(entry)) {
      return fail(`${where} must be an object spec`);
    }
    const spec = entry as Record<string, unknown>;

    if (typeof spec.image !== 'string' || spec.image.trim().length === 0) {
      return fail(`${where}.image must be a non-empty string`);
    }

    // Defaults apply ONLY to genuinely omitted fields (`undefined`).
    // An explicit `null` (hand-edit leftovers, templating artifacts) is
    // a shape error and fails loudly like any other wrong type — the
    // loader must never quietly reinterpret a present-but-wrong value.
    const mountsRaw = spec.mounts === undefined ? [] : spec.mounts;
    if (
      !Array.isArray(mountsRaw) ||
      mountsRaw.some((m) => typeof m !== 'string')
    ) {
      return fail(`${where}.mounts must be an array of strings`);
    }
    const mounts: string[] = [];
    for (const [i, mount] of (mountsRaw as string[]).entries()) {
      const expanded = expandMountVars(mount);
      if (!expanded.ok) {
        return fail(`${where}.mounts[${i}]: ${expanded.error}`);
      }
      // Both sides of the bind must be non-empty so a malformed spec
      // fails HERE with the entry named, not later inside `docker run`
      // with a generic daemon error. A third `:options` segment
      // (`:ro` etc.) is allowed but must be non-empty when present.
      const segments = expanded.value.split(':');
      if (
        segments.length < 2 ||
        segments.length > 3 ||
        segments.some((s) => s.length === 0)
      ) {
        return fail(
          `${where}.mounts[${i}] must be a "hostPath:containerPath[:options]" bind spec with non-empty segments`,
        );
      }
      mounts.push(expanded.value);
    }

    const baseArgs = spec.baseArgs === undefined ? [] : spec.baseArgs;
    if (
      !Array.isArray(baseArgs) ||
      baseArgs.some((a) => typeof a !== 'string')
    ) {
      return fail(`${where}.baseArgs must be an array of strings`);
    }
    const allowedFlags =
      spec.allowedFlags === undefined ? [] : spec.allowedFlags;
    if (
      !Array.isArray(allowedFlags) ||
      allowedFlags.some((f) => typeof f !== 'string')
    ) {
      return fail(`${where}.allowedFlags must be an array of strings`);
    }

    const timeoutMs =
      spec.timeoutMs === undefined ? DEFAULT_TIMEOUT_MS : spec.timeoutMs;
    if (
      typeof timeoutMs !== 'number' ||
      !Number.isFinite(timeoutMs) ||
      timeoutMs <= 0
    ) {
      return fail(`${where}.timeoutMs must be a positive number (ms)`);
    }
    const maxBuffer =
      spec.maxBuffer === undefined ? DEFAULT_MAX_BUFFER : spec.maxBuffer;
    if (
      typeof maxBuffer !== 'number' ||
      !Number.isFinite(maxBuffer) ||
      maxBuffer <= 0
    ) {
      return fail(`${where}.maxBuffer must be a positive number (bytes)`);
    }

    registry[name] = {
      image: spec.image,
      mounts,
      baseArgs: baseArgs as string[],
      allowedFlags: allowedFlags as string[],
      timeoutMs,
      maxBuffer,
    };
  }

  return { ok: true, registry, configPath };
}

/**
 * Expand `${VAR}` placeholders in a mount string from the host env,
 * plus the built-in `HOST_PROJECT_PARENT` (directory containing the
 * project root). Unknown variables are config errors — a silent empty
 * string would produce a mount like `/.audible:/root/.audible` that
 * binds the wrong host path.
 */
function expandMountVars(
  mount: string,
): { ok: true; value: string } | { ok: false; error: string } {
  let error: string | null = null;
  const value = mount.replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_, name) => {
    if (name === 'HOST_PROJECT_PARENT') {
      return path.dirname(process.env.HOST_PROJECT_ROOT || process.cwd());
    }
    const fromEnv = process.env[name];
    if (fromEnv === undefined || fromEnv === '') {
      error = `environment variable \${${name}} is unset — export it (or fix the placeholder) before running this sidecar`;
      return '';
    }
    return fromEnv;
  });
  if (error) return { ok: false, error };
  return { ok: true, value };
}
