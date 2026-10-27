import {
  isCommandError,
  ContainerExitError,
  ContainerOutputProtocolError,
} from './operational-errors.js';
/**
 * Container Runner for NanoClaw
 * Spawns agent execution in containers and handles IPC
 */
import { ChildProcess, spawn } from 'child_process';
import fs from 'fs';
import { StringDecoder } from 'string_decoder';
import { protectIpcClaimMounts } from './ipc-mounts.js';
import path from 'path';

import {
  CONTAINER_IMAGE,
  CONTAINER_MAX_OUTPUT_SIZE,
  CONTAINER_TIMEOUT,
  DATA_DIR,
  GROUPS_DIR,
  IDLE_TIMEOUT,
  ONECLI_URL,
  TIMEZONE,
} from './config.js';
import { resolveGroupFolderPath, resolveGroupIpcPath } from './group-folder.js';
import { logger } from './logger.js';
import {
  CONTAINER_RUNTIME_BIN,
  hostGatewayArgs,
  readonlyMountArgs,
  stopContainer,
} from './container-runtime.js';
import { OneCLI } from '@onecli-sh/sdk';
import { validateAdditionalMounts } from './mount-security.js';
import { RegisteredGroup } from './types.js';

const onecli = new OneCLI({ url: ONECLI_URL });

// Sentinel markers for robust output parsing (must match agent-runner)
const OUTPUT_START_MARKER = '---NANOCLAW_OUTPUT_START---';
const OUTPUT_END_MARKER = '---NANOCLAW_OUTPUT_END---';

export interface ContainerInput {
  prompt: string;
  sessionId?: string;
  groupFolder: string;
  chatJid: string;
  isMain: boolean;
  isScheduledTask?: boolean;
  assistantName?: string;
  script?: string;
}

export interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
  streamText?: string; // Accumulated text for streaming preview
  completion?: 'sdk_turn_completed' | 'script_precheck_completed'; // Never delivery evidence
}

interface VolumeMount {
  hostPath: string;
  containerPath: string;
  readonly: boolean;
}

function buildVolumeMounts(
  group: RegisteredGroup,
  isMain: boolean,
): VolumeMount[] {
  const mounts: VolumeMount[] = [];
  const projectRoot = process.cwd();
  const groupDir = resolveGroupFolderPath(group.folder);

  if (isMain) {
    // Main gets the project root read-only. Writable paths the agent needs
    // (group folder, IPC, .claude/) are mounted separately below.
    // Read-only prevents the agent from modifying host application code
    // (src/, dist/, package.json, etc.) which would bypass the sandbox
    // entirely on next restart.
    mounts.push({
      hostPath: projectRoot,
      containerPath: '/workspace/project',
      readonly: true,
    });

    // Shadow .env so the agent cannot read secrets from the mounted project root.
    // Credentials are injected by the OneCLI gateway, never exposed to containers.
    const envFile = path.join(projectRoot, '.env');
    if (fs.existsSync(envFile)) {
      mounts.push({
        hostPath: '/dev/null',
        containerPath: '/workspace/project/.env',
        readonly: true,
      });
    }

    // Main also gets its group folder as the working directory
    mounts.push({
      hostPath: groupDir,
      containerPath: '/workspace/group',
      readonly: false,
    });
  } else {
    // Other groups only get their own folder
    mounts.push({
      hostPath: groupDir,
      containerPath: '/workspace/group',
      readonly: false,
    });

    // Global memory directory (read-only for non-main)
    // Only directory mounts are supported, not file mounts
    const globalDir = path.join(GROUPS_DIR, 'global');
    if (fs.existsSync(globalDir)) {
      mounts.push({
        hostPath: globalDir,
        containerPath: '/workspace/global',
        readonly: true,
      });
    }
  }

  // Per-group Claude sessions directory (isolated from other groups)
  // Each group gets their own .claude/ to prevent cross-group session access
  const groupSessionsDir = path.join(
    DATA_DIR,
    'sessions',
    group.folder,
    '.claude',
  );
  fs.mkdirSync(groupSessionsDir, { recursive: true });
  const settingsFile = path.join(groupSessionsDir, 'settings.json');
  if (!fs.existsSync(settingsFile)) {
    fs.writeFileSync(
      settingsFile,
      JSON.stringify(
        {
          env: {
            // Enable agent swarms (subagent orchestration)
            // https://code.claude.com/docs/en/agent-teams#orchestrate-teams-of-claude-code-sessions
            CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS: '1',
            // Load CLAUDE.md from additional mounted directories
            // https://code.claude.com/docs/en/memory#load-memory-from-additional-directories
            CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD: '1',
            // Enable Claude's memory feature (persists user preferences between sessions)
            // https://code.claude.com/docs/en/memory#manage-auto-memory
            CLAUDE_CODE_DISABLE_AUTO_MEMORY: '0',
          },
        },
        null,
        2,
      ) + '\n',
    );
  }

  // Sync skills from container/skills/ into each group's .claude/skills/
  const skillsSrc = path.join(process.cwd(), 'container', 'skills');
  const skillsDst = path.join(groupSessionsDir, 'skills');
  if (fs.existsSync(skillsSrc)) {
    for (const skillDir of fs.readdirSync(skillsSrc)) {
      const srcDir = path.join(skillsSrc, skillDir);
      if (!fs.statSync(srcDir).isDirectory()) continue;
      const dstDir = path.join(skillsDst, skillDir);
      fs.cpSync(srcDir, dstDir, { recursive: true });
    }
  }
  mounts.push({
    hostPath: groupSessionsDir,
    containerPath: '/home/node/.claude',
    readonly: false,
  });

  // Per-group IPC namespace: each group gets its own IPC directory
  // This prevents cross-group privilege escalation via IPC
  const groupIpcDir = resolveGroupIpcPath(group.folder);
  fs.mkdirSync(path.join(groupIpcDir, 'messages'), { recursive: true });
  fs.mkdirSync(path.join(groupIpcDir, 'tasks'), { recursive: true });
  fs.mkdirSync(path.join(groupIpcDir, 'input'), { recursive: true });
  mounts.push({
    hostPath: groupIpcDir,
    containerPath: '/workspace/ipc',
    readonly: false,
  });

  // Copy agent-runner source into a per-group writable location so agents
  // can customize it (add tools, change behavior) without affecting other
  // groups. Recompiled on container startup via entrypoint.sh.
  const agentRunnerSrc = path.join(
    projectRoot,
    'container',
    'agent-runner',
    'src',
  );
  const groupAgentRunnerDir = path.join(
    DATA_DIR,
    'sessions',
    group.folder,
    'agent-runner-src',
  );
  if (fs.existsSync(agentRunnerSrc)) {
    const srcIndex = path.join(agentRunnerSrc, 'index.ts');
    const cachedIndex = path.join(groupAgentRunnerDir, 'index.ts');
    const needsCopy =
      !fs.existsSync(groupAgentRunnerDir) ||
      !fs.existsSync(cachedIndex) ||
      (fs.existsSync(srcIndex) &&
        fs.statSync(srcIndex).mtimeMs > fs.statSync(cachedIndex).mtimeMs);
    if (needsCopy) {
      fs.cpSync(agentRunnerSrc, groupAgentRunnerDir, { recursive: true });
    }
  }
  mounts.push({
    hostPath: groupAgentRunnerDir,
    containerPath: '/app/src',
    readonly: false,
  });

  // Additional mounts validated against external allowlist (tamper-proof from containers)
  if (group.containerConfig?.additionalMounts) {
    const validatedMounts = validateAdditionalMounts(
      group.containerConfig.additionalMounts,
      group.name,
      isMain,
    );
    mounts.push(...validatedMounts);
  }

  return mounts;
}

async function buildContainerArgs(
  mounts: VolumeMount[],
  containerName: string,
  agentIdentifier?: string,
): Promise<string[]> {
  const args: string[] = ['run', '-i', '--rm', '--name', containerName];

  // Pass host timezone so container's local time matches the user's
  args.push('-e', `TZ=${TIMEZONE}`);

  // OneCLI gateway handles credential injection — containers never see real secrets.
  // The gateway intercepts HTTPS traffic and injects API keys or OAuth tokens.
  const onecliApplied = await onecli.applyContainerConfig(args, {
    addHostMapping: false, // Nanoclaw already handles host gateway
    agent: agentIdentifier,
  });
  if (onecliApplied) {
    logger.info({ containerName }, 'OneCLI gateway config applied');
  } else {
    logger.warn(
      { containerName },
      'OneCLI gateway not reachable — container will have no credentials',
    );
  }

  // Runtime-specific args for host gateway resolution
  args.push(...hostGatewayArgs());

  // Run as host user so bind-mounted files are accessible.
  // Skip when running as root (uid 0), as the container's node user (uid 1000),
  // or when getuid is unavailable (native Windows without WSL).
  const hostUid = process.getuid?.();
  const hostGid = process.getgid?.();
  if (hostUid != null && hostUid !== 0 && hostUid !== 1000) {
    args.push('--user', `${hostUid}:${hostGid}`);
    args.push('-e', 'HOME=/home/node');
  }

  for (const mount of mounts) {
    if (mount.readonly) {
      args.push(...readonlyMountArgs(mount.hostPath, mount.containerPath));
    } else {
      args.push('-v', `${mount.hostPath}:${mount.containerPath}`);
    }
  }

  args.push(CONTAINER_IMAGE);

  return args;
}

function parseContainerOutput(json: string): ContainerOutput {
  const output: unknown = JSON.parse(json);
  if (!output || typeof output !== 'object' || Array.isArray(output)) {
    throw new SyntaxError('Container output must be an object');
  }
  const fields = output as Record<string, unknown>;
  if (
    (fields.status !== 'success' && fields.status !== 'error') ||
    (fields.result !== null && typeof fields.result !== 'string')
  )
    throw new SyntaxError('Container output has invalid status or result');
  for (const field of ['newSessionId', 'error', 'streamText']) {
    if (fields[field] !== undefined && typeof fields[field] !== 'string')
      throw new SyntaxError(`Container output ${field} must be a string`);
  }
  if (
    fields.completion !== undefined &&
    ((fields.completion !== 'sdk_turn_completed' &&
      fields.completion !== 'script_precheck_completed') ||
      fields.status !== 'success' ||
      fields.streamText !== undefined)
  )
    throw new SyntaxError('Container output has an invalid SDK completion');
  return fields as unknown as ContainerOutput;
}

export async function runContainerAgent(
  group: RegisteredGroup,
  input: ContainerInput,
  onProcess: (proc: ChildProcess, containerName: string) => void,
  onOutput?: (output: ContainerOutput) => Promise<void>,
  outputFailure?: Promise<{ error: unknown }>,
): Promise<ContainerOutput> {
  const startTime = Date.now();

  const groupDir = resolveGroupFolderPath(group.folder);
  fs.mkdirSync(groupDir, { recursive: true });

  const mounts = protectIpcClaimMounts(buildVolumeMounts(group, input.isMain));
  const safeName = group.folder.replace(/[^a-zA-Z0-9-]/g, '-');
  const containerName = `nanoclaw-${safeName}-${Date.now()}`;
  // Main group uses the default OneCLI agent; others use their own agent.
  const agentIdentifier = input.isMain
    ? undefined
    : group.folder.toLowerCase().replace(/_/g, '-');
  const containerArgs = await buildContainerArgs(
    mounts,
    containerName,
    agentIdentifier,
  );

  logger.debug(
    {
      group: group.name,
      containerName,
      mounts: mounts.map(
        (m) =>
          `${m.hostPath} -> ${m.containerPath}${m.readonly ? ' (ro)' : ''}`,
      ),
      containerArgs: containerArgs.join(' '),
    },
    'Container mount configuration',
  );

  logger.info(
    {
      group: group.name,
      containerName,
      mountCount: mounts.length,
      isMain: input.isMain,
    },
    'Spawning container agent',
  );

  const logsDir = path.join(groupDir, 'logs');
  fs.mkdirSync(logsDir, { recursive: true });

  return new Promise((resolve, reject) => {
    const container = spawn(CONTAINER_RUNTIME_BIN, containerArgs, {
      stdio: ['pipe', 'pipe', 'pipe'],
    });

    let stdout = '';
    const stdoutDecoder = new StringDecoder('utf8');
    let stderr = '';
    let stdoutTruncated = false;
    let stderrTruncated = false;

    // Streaming output: parse OUTPUT_START/END marker pairs as they arrive
    // Retain raw frame bytes until both markers arrive, so UTF-8 characters
    // split across native stdout chunks are decoded exactly once.
    let parseBuffer = Buffer.alloc(0);
    let newSessionId: string | undefined;
    let outputProtocolError = false;
    const hasInvalidFrame = () =>
      outputProtocolError || parseBuffer.includes(OUTPUT_START_MARKER);
    let outputChain = Promise.resolve();
    let timeout: ReturnType<typeof setTimeout> | undefined;
    let callbackFailed = false;
    let failurePending = false;
    let containerClosed = false;
    let spawnErrored = false;
    let nativeTerminalFailure: ContainerExitError | undefined;
    const fail = (callbackError: unknown) => {
      const error = nativeTerminalFailure
        ? new AggregateError(
            [callbackError, nativeTerminalFailure],
            'Container output callback and native exit failed; inspect both before retrying.',
            { cause: callbackError },
          )
        : callbackError;
      if (failurePending) return;
      failurePending = true;
      callbackFailed = true;
      clearTimeout(timeout);
      if (containerClosed || spawnErrored) {
        reject(error);
        return;
      }
      Promise.resolve()
        .then(() => stopContainer(containerName))
        .then(
          () => reject(error),
          (cleanupError: unknown) =>
            reject(
              new AggregateError(
                [error, cleanupError],
                'Container callback and owned-container cleanup failed; inspect the container before retrying.',
                { cause: error },
              ),
            ),
        );
    };
    // Promise executors preserve synchronous event ordering while forwarding a
    // thrown callback defect to this run's returned Promise, rather than the host.
    const runGuarded = (callback: () => void) => {
      if (callbackFailed) return;
      void new Promise<void>((done) => {
        let completed = false;
        try {
          callback();
          completed = true;
          done();
        } finally {
          if (!completed) callbackFailed = true;
        }
      }).then(undefined, fail);
    };
    const afterOutput = (callback: () => void) => {
      void outputChain
        .then(() => {
          if (!callbackFailed) callback();
        })
        .then(undefined, fail);
    };

    container.stdout.on('data', (data) =>
      runGuarded(() => {
        const chunk = stdoutDecoder.write(data);

        // Always accumulate for logging
        if (!stdoutTruncated) {
          const remaining = CONTAINER_MAX_OUTPUT_SIZE - stdout.length;
          if (chunk.length > remaining) {
            stdout += chunk.slice(0, remaining);
            stdoutTruncated = true;
            logger.warn(
              { group: group.name, size: stdout.length },
              'Container stdout truncated due to size limit',
            );
          } else {
            stdout += chunk;
          }
        }

        // Stream-parse for output markers
        if (onOutput) {
          const bytes = Buffer.isBuffer(data) ? data : Buffer.from(data);
          if (parseBuffer.length + bytes.length > CONTAINER_MAX_OUTPUT_SIZE) {
            parseBuffer = Buffer.alloc(0);
            throw new ContainerOutputProtocolError(
              'Agent output exceeded the bounded frame parser; retain undelivered input and inspect the agent protocol before retrying.',
            );
          }
          parseBuffer = Buffer.concat([parseBuffer, bytes]);
          let startIdx: number;
          while ((startIdx = parseBuffer.indexOf(OUTPUT_START_MARKER)) !== -1) {
            const endIdx = parseBuffer.indexOf(OUTPUT_END_MARKER, startIdx);
            if (endIdx === -1) break; // Incomplete pair, wait for more data

            const jsonStr = parseBuffer
              .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
              .toString('utf8')
              .trim();
            parseBuffer = parseBuffer.slice(endIdx + OUTPUT_END_MARKER.length);

            try {
              const parsed = parseContainerOutput(jsonStr);
              if (
                parsed.completion === 'script_precheck_completed' &&
                (!input.script || !input.isScheduledTask)
              )
                throw new SyntaxError(
                  'Script completion requires a scheduled precheck',
                );
              if (parsed.newSessionId) {
                newSessionId = parsed.newSessionId;
              }
              hadStreamingOutput = true;
              const responseActivity =
                parsed.completion !== undefined ||
                parsed.streamText !== undefined ||
                parsed.result !== null ||
                parsed.status === 'error';
              const markerSequence = responseActivity
                ? ++latestMarkerSequence
                : latestMarkerSequence;
              if (responseActivity) completedTerminalOutput = false;
              // Activity detected — reset the hard timeout
              resetTimeout();
              // Call onOutput for all markers (including null results)
              // so idle timers start even for "silent" query completions.
              outputChain = outputChain.then(async () => {
                if (callbackFailed) return;
                await onOutput(parsed);
                if (
                  markerSequence === latestMarkerSequence &&
                  parsed.status === 'success' &&
                  (parsed.completion !== undefined ||
                    (parsed.result !== null && parsed.result.trim()))
                )
                  completedTerminalOutput = true;
              });
              void outputChain.then(undefined, fail);
            } catch (err) {
              if (!(err instanceof SyntaxError)) throw err;
              outputProtocolError = true;
              logger.warn(
                { group: group.name },
                'Failed to parse streamed output chunk',
              );
            }
          }
        }
      }),
    );

    container.stderr.on('data', (data) =>
      runGuarded(() => {
        const chunk = data.toString();
        const lines = chunk.trim().split('\n');
        for (const line of lines) {
          if (line)
            logger.debug(
              { container: group.folder, length: line.length },
              'Container stderr activity',
            );
        }
        // Don't reset timeout on stderr — SDK writes debug logs continuously.
        // Timeout only resets on actual output (OUTPUT_MARKER in stdout).
        if (stderrTruncated) return;
        const remaining = CONTAINER_MAX_OUTPUT_SIZE - stderr.length;
        if (chunk.length > remaining) {
          stderr += chunk.slice(0, remaining);
          stderrTruncated = true;
          logger.warn(
            { group: group.name, size: stderr.length },
            'Container stderr truncated due to size limit',
          );
        } else {
          stderr += chunk;
        }
      }),
    );

    let timedOut = false;
    let hadStreamingOutput = false;
    let latestMarkerSequence = 0;
    let completedTerminalOutput = false;
    const configTimeout = group.containerConfig?.timeout || CONTAINER_TIMEOUT;
    // Grace period: hard timeout must be at least IDLE_TIMEOUT + 30s so the
    // graceful _close sentinel has time to trigger before the hard kill fires.
    const timeoutMs = Math.max(configTimeout, IDLE_TIMEOUT + 30_000);

    const killOnTimeout = () => {
      timedOut = true;
      logger.error(
        { group: group.name, containerName },
        'Container timeout, stopping gracefully',
      );
      try {
        stopContainer(containerName);
      } catch (err) {
        if (!isCommandError(err)) throw err;
        logger.warn(
          { group: group.name, containerName, err },
          'Graceful stop failed, force killing',
        );
        container.kill('SIGKILL');
      }
    };

    timeout = setTimeout(() => runGuarded(killOnTimeout), timeoutMs);

    // Reset the timeout whenever there's activity (streaming output)
    const resetTimeout = () => {
      clearTimeout(timeout);
      timeout = setTimeout(() => runGuarded(killOnTimeout), timeoutMs);
    };

    container.on('close', (code) => {
      containerClosed = true;
      if (code !== 0) nativeTerminalFailure = new ContainerExitError(code);
      runGuarded(() => {
        clearTimeout(timeout);
        const duration = Date.now() - startTime;

        if (timedOut) {
          const ts = new Date().toISOString().replace(/[:.]/g, '-');
          const timeoutLog = path.join(logsDir, `container-${ts}.log`);
          fs.writeFileSync(
            timeoutLog,
            [
              `=== Container Run Log (TIMEOUT) ===`,
              `Timestamp: ${new Date().toISOString()}`,
              `Group: ${group.name}`,
              `Container: ${containerName}`,
              `Duration: ${duration}ms`,
              `Exit Code: ${code}`,
              `Had Streaming Output: ${hadStreamingOutput}`,
            ].join('\n'),
          );

          // A settled result or actual SDK completion permits idle cleanup.
          // Completion does not assert that an IPC message reached its channel.
          afterOutput(() => {
            if (hasInvalidFrame()) {
              resolve({
                status: 'error',
                result: null,
                newSessionId,
                error:
                  'Agent emitted malformed or unfinished output before timeout; retain undelivered input and inspect its output protocol.',
              });
              return;
            }
            if (completedTerminalOutput) {
              logger.info(
                { group: group.name, containerName, duration, code },
                'Container timed out after terminal output (idle cleanup)',
              );
              resolve({ status: 'success', result: null, newSessionId });
            } else {
              logger.error(
                { group: group.name, containerName, duration, code },
                'Container timed out without completed terminal output',
              );
              resolve({
                status: 'error',
                result: null,
                error: `Container timed out after ${configTimeout}ms without a completed terminal response`,
              });
            }
          });
          return;
        }

        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const logFile = path.join(logsDir, `container-${timestamp}.log`);
        const logLines = [
          '=== Container Run Metadata ===',
          `Timestamp: ${new Date().toISOString()}`,
          `Duration: ${duration}ms`,
          `Exit Code: ${code}`,
          `Prompt length: ${input.prompt.length} chars`,
          `Script length: ${input.script?.length ?? 0} chars`,
          `Stdout length: ${stdout.length} chars`,
          `Stderr length: ${stderr.length} chars`,
          `Stdout Truncated: ${stdoutTruncated}`,
          `Stderr Truncated: ${stderrTruncated}`,
        ];

        fs.writeFileSync(logFile, logLines.join('\n'));
        logger.debug({ logFile }, 'Container metadata log written');

        if (code !== 0) {
          logger.error(
            {
              group: group.name,
              code,
              duration,
              stderrLength: stderr.length,
              stdoutLength: stdout.length,
              logFile,
            },
            'Container exited with error',
          );

          afterOutput(() =>
            resolve({
              status: 'error',
              result: null,
              error: `Container exited with code ${code}; inspect native runtime health before retrying.`,
            }),
          );
          return;
        }

        // Streaming mode: wait for output chain to settle, return completion marker
        if (onOutput) {
          if (hasInvalidFrame()) {
            afterOutput(() =>
              resolve({
                status: 'error',
                result: null,
                newSessionId,
                error:
                  'Agent emitted invalid output; retain the input and restart the agent after inspecting its output protocol.',
              }),
            );
            return;
          }
          afterOutput(() => {
            if (!completedTerminalOutput) {
              resolve({
                status: 'error',
                result: null,
                newSessionId,
                error:
                  'Agent exited without a completed terminal response; retain undelivered input and inspect the output protocol.',
              });
              return;
            }
            logger.info(
              { group: group.name, duration, newSessionId },
              'Container completed (streaming mode)',
            );
            resolve({
              status: 'success',
              result: null,
              newSessionId,
            });
          });
          return;
        }

        // Legacy mode: parse the last output marker pair from accumulated stdout
        try {
          // Extract JSON between sentinel markers for robust parsing
          const startIdx = stdout.indexOf(OUTPUT_START_MARKER);
          const endIdx = stdout.indexOf(OUTPUT_END_MARKER);

          let jsonLine: string;
          if (startIdx !== -1 && endIdx !== -1 && endIdx > startIdx) {
            jsonLine = stdout
              .slice(startIdx + OUTPUT_START_MARKER.length, endIdx)
              .trim();
          } else {
            // Fallback: last non-empty line (backwards compatibility)
            const lines = stdout.trim().split('\n');
            jsonLine = lines[lines.length - 1];
          }

          const output = parseContainerOutput(jsonLine);

          logger.info(
            {
              group: group.name,
              duration,
              status: output.status,
              hasResult: !!output.result,
            },
            'Container completed',
          );

          afterOutput(() => resolve(output));
        } catch (err) {
          if (!(err instanceof SyntaxError)) throw err;
          logger.error(
            {
              group: group.name,
              stdoutLength: stdout.length,
              stderrLength: stderr.length,
            },
            'Failed to parse container output',
          );

          afterOutput(() =>
            resolve({
              status: 'error',
              result: null,
              error:
                'Failed to parse container output; inspect the agent output protocol before retrying.',
            }),
          );
        }
      });
    });

    container.on('error', (err) => {
      spawnErrored = true;
      runGuarded(() => {
        clearTimeout(timeout);
        const nativeCode =
          'code' in err &&
          typeof err.code === 'string' &&
          ['ENOENT', 'EACCES', 'EPERM', 'EAGAIN', 'ENOMEM', 'EIO'].includes(
            err.code,
          )
            ? err.code
            : 'unknown';
        logger.error(
          { group: group.name, containerName, nativeCode },
          'Container spawn error',
        );
        afterOutput(() =>
          resolve({
            status: 'error',
            result: null,
            error: `Container spawn error (${nativeCode}); verify the native container runtime before retrying.`,
          }),
        );
      });
    });
    if (outputFailure)
      void outputFailure.then(({ error }) => fail(error)).then(undefined, fail);
    container.stdin.on('error', fail);
    runGuarded(() => {
      onProcess(container, containerName);
      container.stdin.write(JSON.stringify(input));
      container.stdin.end();
    });
  });
}

export function writeTasksSnapshot(
  groupFolder: string,
  isMain: boolean,
  tasks: Array<{
    id: string;
    groupFolder: string;
    prompt: string;
    script?: string | null;
    schedule_type: string;
    schedule_value: string;
    status: string;
    next_run: string | null;
  }>,
): void {
  // Write filtered tasks to the group's IPC directory
  const groupIpcDir = resolveGroupIpcPath(groupFolder);
  fs.mkdirSync(groupIpcDir, { recursive: true });

  // Main sees all tasks, others only see their own
  const filteredTasks = isMain
    ? tasks
    : tasks.filter((t) => t.groupFolder === groupFolder);

  const tasksFile = path.join(groupIpcDir, 'current_tasks.json');
  fs.writeFileSync(tasksFile, JSON.stringify(filteredTasks, null, 2));
}

export interface AvailableGroup {
  jid: string;
  name: string;
  lastActivity: string;
  isRegistered: boolean;
}

/**
 * Write available groups snapshot for the container to read.
 * Only main group can see all available groups (for activation).
 * Non-main groups only see their own registration status.
 */
export function writeGroupsSnapshot(
  groupFolder: string,
  isMain: boolean,
  groups: AvailableGroup[],
  _registeredJids: Set<string>,
): void {
  const groupIpcDir = resolveGroupIpcPath(groupFolder);
  fs.mkdirSync(groupIpcDir, { recursive: true });

  // Main sees all groups; others see nothing (they can't activate groups)
  const visibleGroups = isMain ? groups : [];

  const groupsFile = path.join(groupIpcDir, 'available_groups.json');
  fs.writeFileSync(
    groupsFile,
    JSON.stringify(
      {
        groups: visibleGroups,
        lastSync: new Date().toISOString(),
      },
      null,
      2,
    ),
  );
}
