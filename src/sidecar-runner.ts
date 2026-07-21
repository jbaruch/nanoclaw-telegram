import { execFile } from 'child_process';

import { logger } from './logger.js';
import { loadSidecarRegistry } from './sidecar-config.js';

/**
 * A privileged docker sidecar the host can run on a plugin's behalf. The
 * security-sensitive surface — which image runs and which host paths it
 * bind-mounts — lives in trusted host config, NEVER in the IPC payload.
 *
 * Named-sidecar registry (#750, part of #741): a plugin references an
 * entry by NAME and may only append allowlisted flags, mirroring
 * OneCLI's named-credential model — a compromised container cannot
 * request `-v /:/…` or mount the host `.env`. Since #850 the entries
 * themselves are DATA, loaded per call from `config/sidecars.json` (see
 * `sidecar-config.ts`), so adding a sidecar is a host-config edit, not
 * a TS change, and no personal NAS path lives in committed source.
 */
export interface SidecarSpec {
  image: string;
  /** Host→container bind mounts, each `hostPath:containerPath`. */
  mounts: string[];
  /** Fixed CLI args always passed to the image (e.g. output format). */
  baseArgs: string[];
  /** Bare flags a plugin may append, allowlisted per sidecar. */
  allowedFlags: readonly string[];
  timeoutMs: number;
  maxBuffer: number;
}

export interface RunSidecarRequest {
  name: string;
  flags?: string[];
}

/**
 * Run a registered privileged sidecar and resolve its result payload. Never
 * rejects for an operational failure; only a genuine bug rejects.
 *
 * Result shapes:
 * - Unknown sidecar name or a flag outside the entry's allowlist → an
 *   `{ error }` envelope (docker is never invoked).
 * - Docker ran and stdout is JSON → that parsed object, verbatim. On a
 *   non-zero exit the sidecar's own payload is preserved and an `exec_error`
 *   field is added alongside it (the #625 partial-success contract), without
 *   overwriting a script-emitted top-level `error`.
 * - Docker ran but stdout is not JSON → an `{ error, raw, logs }` envelope.
 *
 * Only a non-`SyntaxError` thrown while parsing stdout (a real defect, not a
 * malformed-output case) rejects; the `run_sidecar` IPC handler's
 * outer-boundary `.catch()` turns that into a result envelope.
 */
export async function runSidecar(
  req: RunSidecarRequest,
): Promise<Record<string, unknown>> {
  const loaded = loadSidecarRegistry();
  if (!loaded.ok) {
    // Fail loudly and actionably (#850): a present-but-broken config
    // must never degrade into a silent empty registry.
    logger.error(
      { name: req.name, configPath: loaded.configPath, error: loaded.error },
      'run_sidecar: sidecar config invalid',
    );
    return { error: `Sidecar config invalid: ${loaded.error}` };
  }
  const registry = loaded.registry;
  const spec = registry[req.name];
  if (!spec) {
    logger.warn({ name: req.name }, 'run_sidecar: unknown sidecar');
    return {
      error: `Unknown sidecar "${req.name}". Registered: ${Object.keys(registry).join(', ') || '(none)'}. Add it to ${loaded.configPath} — see config/README.md.`,
    };
  }

  const flags = req.flags ?? [];
  const disallowed = flags.filter((f) => !spec.allowedFlags.includes(f));
  if (disallowed.length > 0) {
    logger.warn(
      { name: req.name, disallowed },
      'run_sidecar: disallowed flag(s)',
    );
    return {
      error: `Flag(s) ${JSON.stringify(disallowed)} not permitted for sidecar "${req.name}" (allowed: ${spec.allowedFlags.join(', ') || 'none'}).`,
    };
  }

  const dockerArgs = [
    'run',
    '--rm',
    ...spec.mounts.flatMap((m) => ['-v', m]),
    spec.image,
    ...spec.baseArgs,
    ...flags,
  ];

  logger.info({ name: req.name, flags }, 'Running sidecar');

  const { error, stdout, stderr } = await execDocker(dockerArgs, {
    cwd: process.cwd(),
    env: {
      PATH: process.env.PATH || '/usr/bin:/bin',
      HOME: process.env.HOME || '/root',
    },
    timeout: spec.timeoutMs,
    maxBuffer: spec.maxBuffer,
  });

  if (error) {
    logger.error(
      { name: req.name, error: error.message, stderr: stderr?.slice(-2000) },
      'sidecar failed',
    );
  } else {
    logger.info(
      { name: req.name, stdoutLen: stdout.length },
      'sidecar completed',
    );
  }

  // Parse stdout first (partial-success relay). JSON.parse only throws
  // SyntaxError for non-JSON stdout, which the raw-envelope fallback below
  // handles. Any other throw is a real bug and must propagate: this runs in
  // the async body so it rejects runSidecar's promise, and the run_sidecar
  // IPC handler's outer-boundary `.catch()` turns that into a result
  // envelope so the MCP caller never hangs.
  let parsed: Record<string, unknown> | null = null;
  try {
    const value = JSON.parse(stdout);
    parsed =
      value !== null && typeof value === 'object' && !Array.isArray(value)
        ? (value as Record<string, unknown>)
        : null;
  } catch (e) {
    if (!(e instanceof SyntaxError)) throw e;
    parsed = null;
  }

  if (parsed !== null) {
    if (stderr) parsed.logs = stderr.slice(-2000);
    // Don't overwrite a script-emitted top-level `error`; surface the exec
    // error in a sibling field so both signals survive.
    if (error) parsed.exec_error = error.message;
    return parsed;
  }
  return {
    error: error?.message,
    raw: stdout,
    logs: stderr?.slice(-2000),
  };
}

interface DockerResult {
  error: (Error & { code?: number | string; killed?: boolean }) | null;
  stdout: string;
  stderr: string;
}

/**
 * Thin promise wrapper around `execFile('docker', …)` that never rejects. A
 * non-zero exit or timeout is an operational outcome relayed via the `error`
 * field alongside the child's stdout/stderr (so `runSidecar` can apply the
 * #625 partial-success relay); the executor has no throwing path of its own.
 */
function execDocker(
  args: string[],
  options: {
    cwd: string;
    env: NodeJS.ProcessEnv;
    timeout: number;
    maxBuffer: number;
  },
): Promise<DockerResult> {
  return new Promise((resolve) => {
    execFile('docker', args, options, (error, stdout, stderr) => {
      resolve({ error: error as DockerResult['error'], stdout, stderr });
    });
  });
}
