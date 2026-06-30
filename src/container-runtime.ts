/**
 * Container runtime abstraction for NanoClaw.
 * All runtime-specific logic lives here so swapping runtimes means changing one file.
 */
import { execSync, spawnSync } from 'child_process';
import fs from 'fs';
import os from 'os';

import { logger } from './logger.js';

/** The container runtime binary name. */
export const CONTAINER_RUNTIME_BIN = 'docker';

/** Hostname containers use to reach the host machine. */
export const CONTAINER_HOST_GATEWAY = 'host.docker.internal';

/**
 * Address the credential proxy binds to.
 * Docker Desktop (macOS): 127.0.0.1 — the VM routes host.docker.internal to loopback.
 * Docker (Linux): bind to the docker0 bridge IP so only containers can reach it,
 *   falling back to 0.0.0.0 if the interface isn't found.
 */
export const PROXY_BIND_HOST =
  process.env.CREDENTIAL_PROXY_HOST || detectProxyBindHost();

function detectProxyBindHost(): string {
  if (os.platform() === 'darwin') return '127.0.0.1';

  // WSL uses Docker Desktop (same VM routing as macOS) — loopback is correct.
  if (fs.existsSync('/proc/sys/fs/binfmt_misc/WSLInterop')) return '127.0.0.1';

  // Bare-metal Linux: bind to the docker0 bridge IP instead of 0.0.0.0
  const ifaces = os.networkInterfaces();
  const docker0 = ifaces['docker0'];
  if (docker0) {
    const ipv4 = docker0.find((a) => a.family === 'IPv4');
    if (ipv4) return ipv4.address;
  }
  return '0.0.0.0';
}

/** CLI args needed for the container to resolve the host gateway. */
export function hostGatewayArgs(): string[] {
  // On Linux, host.docker.internal isn't built-in — add it explicitly
  if (os.platform() === 'linux') {
    return ['--add-host=host.docker.internal:host-gateway'];
  }
  return [];
}

/** Returns CLI args for a readonly bind mount. */
export function readonlyMountArgs(
  hostPath: string,
  containerPath: string,
): string[] {
  return ['-v', `${hostPath}:${containerPath}:ro`];
}

/** Stop a container by name. Tries graceful stop first, then SIGKILL. */
export function stopContainer(name: string): void {
  const stop = spawnSync(CONTAINER_RUNTIME_BIN, ['stop', '-t', '1', name], {
    stdio: 'pipe',
    timeout: 10_000,
  });
  if (stop.status !== 0) {
    // Graceful stop failed — force kill
    spawnSync(CONTAINER_RUNTIME_BIN, ['kill', name], {
      stdio: 'pipe',
      timeout: 5_000,
    });
  }
}

/** Ensure the container runtime (Docker) is running. */
export function ensureContainerRuntimeRunning(): void {
  try {
    execSync(`${CONTAINER_RUNTIME_BIN} info`, {
      stdio: 'pipe',
      timeout: 10000,
    });
    logger.debug('Container runtime already running');
  } catch {
    logger.error('Docker is not running or not installed');
    console.error(
      '\n╔════════════════════════════════════════════════════════════════╗',
    );
    console.error(
      '║  FATAL: Docker is not running                                  ║',
    );
    console.error(
      '║                                                                ║',
    );
    console.error(
      '║  Agents cannot run without Docker. To fix:                     ║',
    );
    console.error(
      '║  1. Start Docker Desktop (macOS) or dockerd (Linux)            ║',
    );
    console.error(
      '║  2. Restart NanoClaw                                           ║',
    );
    console.error(
      '╚════════════════════════════════════════════════════════════════╝\n',
    );
    throw new Error('Container runtime is required but failed to start');
  }
}

/**
 * Persistent infrastructure containers that share the `nanoclaw-` prefix
 * but are NOT per-agent orphans — `cleanupOrphans` must never kill them.
 *
 * The #609 LiteLLM gateway runs as a separate UGOS Pro compose project
 * `nanoclaw-litellm` with a single service `nanoclaw-litellm`, so docker
 * names its container `nanoclaw-litellm-nanoclaw-litellm-<idx>`
 * (project + service + replica index). It bakes `ANTHROPIC_API_KEY` via
 * compose `${VAR}` interpolation at create time and does NOT auto-heal
 * from a `docker kill` (`docker kill` bypasses `restart: always`), so
 * killing it on an orchestrator boot silently degrades the credential
 * path until a manual `docker compose up -d`.
 *
 * Match the full project+service+index shape (index left flexible for a
 * recreate/scale bump): a bare `^nanoclaw-litellm$` anchor misses the
 * compose-suffixed name, and a loose `^nanoclaw-litellm` prefix would
 * wrongly exempt a per-group agent whose slug starts with `litellm`
 * (e.g. `nanoclaw-litellm-fans-<ts>`). Lock-step with the deploy-side
 * predicate in `scripts/exclude-infra-containers.sh` (the awk
 * `^nanoclaw-litellm-nanoclaw-litellm-[0-9]+$` negation) — if a second
 * service is ever added to the gateway's compose project, extend both.
 */
const INFRA_CONTAINER_RE = /^nanoclaw-litellm-nanoclaw-litellm-[0-9]+$/;

function isInfraContainer(name: string): boolean {
  return INFRA_CONTAINER_RE.test(name);
}

/**
 * Kill orphaned NanoClaw containers from previous runs.
 *
 * `skipNames` (#213): names the caller has identified as intentional
 * handoffs from a graceful shutdown — they're still doing useful
 * work and should NOT be killed. Names not in the skip set are
 * treated as genuine orphans (crashed-orchestrator leftovers) and
 * stopped as before. An empty / undefined skip set means "no
 * handoff, kill everything" — the pre-#213 behavior, which is the
 * right safety default when no graceful-shutdown marker was found.
 *
 * Persistent infrastructure containers (see `INFRA_CONTAINER_RE`) are
 * excluded unconditionally — they are not agent orphans and must
 * survive every orchestrator restart.
 */
export function cleanupOrphans(skipNames?: ReadonlySet<string>): void {
  try {
    const result = spawnSync(
      CONTAINER_RUNTIME_BIN,
      ['ps', '--format', '{{.Names}}'],
      { stdio: ['pipe', 'pipe', 'pipe'], encoding: 'utf-8' },
    );
    const allNanoclaw = (result.stdout || '')
      .split('\n')
      .map((n) => n.trim())
      .filter((n) => n.startsWith('nanoclaw-') && !isInfraContainer(n));
    const adopted = skipNames
      ? allNanoclaw.filter((n) => skipNames.has(n))
      : [];
    const orphans = skipNames
      ? allNanoclaw.filter((n) => !skipNames.has(n))
      : allNanoclaw;
    for (const name of orphans) {
      try {
        stopContainer(name);
      } catch {
        /* already stopped */
      }
    }
    if (adopted.length > 0) {
      logger.info(
        { count: adopted.length, names: adopted },
        'Adopted detached containers from graceful shutdown (not killed)',
      );
    }
    if (orphans.length > 0) {
      logger.info(
        { count: orphans.length, names: orphans },
        'Stopped orphaned containers',
      );
    }
  } catch (err) {
    logger.warn({ err }, 'Failed to clean up orphaned containers');
  }
}
