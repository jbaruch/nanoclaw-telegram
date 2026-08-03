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

  // Preflight the image (#895). Sidecar images are built locally and never
  // pushed, so once one is gone `docker run` falls back to a registry pull
  // and fails as "pull access denied … may require 'docker login'". That
  // message sends the operator hunting for credentials for an image that
  // was never meant to be pulled. Checking first reports the real problem
  // and its fix instead.
  const preflight = await inspectImage(spec.image);
  if (preflight.state === 'missing') {
    logger.error(
      { name: req.name, image: spec.image },
      'run_sidecar: image missing on host',
    );
    return {
      error:
        `Sidecar image "${spec.image}" is not present on this host. It is built locally and never pushed, ` +
        `so a \`docker system prune\` or host reprovision removes it permanently. ` +
        `Rebuild with \`./scripts/deploy.sh\` (step 2a-bis builds every container/*/build.sh), ` +
        `or directly via \`./container/${req.name}/build.sh\` if that path exists.`,
    };
  }
  if (preflight.state === 'docker-unavailable') {
    // Docker itself is broken (not on PATH, daemon down, inspect timed out).
    // Sending the operator to a build script would waste their time on the
    // wrong problem, and the early return means the real docker error would
    // otherwise never surface — so relay it verbatim.
    logger.error(
      { name: req.name, image: spec.image, error: preflight.message },
      'run_sidecar: docker unavailable during image preflight',
    );
    return {
      error:
        `Could not check for sidecar image "${spec.image}" — docker itself did not respond: ${preflight.message}. ` +
        `This is a docker problem on the host, not a missing image; the sidecar was not run.`,
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

/** Bounds for the preflight probe — small; it prints one JSON blob. */
const IMAGE_INSPECT_TIMEOUT_MS = 10_000;
const IMAGE_INSPECT_MAX_BUFFER = 1024 * 1024;

type PreflightResult =
  | { state: 'present' }
  | { state: 'missing' }
  | { state: 'docker-unavailable'; message: string };

/**
 * Look `image` up in the host daemon's local image store (#895).
 *
 * `docker image inspect` never contacts a registry, so unlike `docker run`
 * it cannot be fooled into reporting a pull failure for an image that was
 * only ever built locally — the misleading "pull access denied" in #895.
 *
 * A non-zero exit is NOT uniformly "missing". Docker being absent from PATH,
 * a stopped daemon, or a timed-out probe all exit non-zero too, and since
 * this preflight returns early, treating those as "missing" would send the
 * operator to a build script for a problem the build script cannot fix — and
 * bury the real docker error. So the absent-image signature is matched
 * explicitly and everything else is relayed as a docker fault.
 */
async function inspectImage(image: string): Promise<PreflightResult> {
  const { error, stderr } = await execDocker(['image', 'inspect', image], {
    cwd: process.cwd(),
    env: {
      PATH: process.env.PATH || '/usr/bin:/bin',
      HOME: process.env.HOME || '/root',
    },
    // A short bound so a wedged daemon can't hold the IPC caller open for
    // the sidecar's full budget (10 minutes for audible-backup).
    timeout: IMAGE_INSPECT_TIMEOUT_MS,
    maxBuffer: IMAGE_INSPECT_MAX_BUFFER,
  });
  if (error === null) return { state: 'present' };
  // Docker's absent-image wording across versions: "No such image",
  // "No such object". Both arrive on stderr with exit 1.
  if (/no such (image|object)/i.test(stderr)) return { state: 'missing' };
  return {
    state: 'docker-unavailable',
    message: (stderr.trim() || error.message).slice(-500),
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
