import fs from 'fs';
import path from 'path';

import { logger } from './logger.js';

export interface SessionArtifactRetentionConfig {
  enabled: boolean;
  maxToolResultBytes: number;
  minToolResultAgeMs: number;
  keepRecentImages: number;
}

export interface SessionArtifactRetentionResult {
  transcriptPath: string | null;
  rewritten: boolean;
  imageBlocksReplaced: number;
  toolResultRefsReplaced: number;
  toolResultFilesDeleted: number;
}

const DEFAULT_MAX_TOOL_RESULT_BYTES = 64 * 1024;
const DEFAULT_MIN_TOOL_RESULT_AGE_MS = 6 * 60 * 60 * 1000;
const DEFAULT_KEEP_RECENT_IMAGES = 2;
// Candidate matcher only. Every match is re-resolved and constrained under
// `<projectDir>/tool-results/` by resolveToolResultRef before any rewrite or
// unlink happens, so incidental text outside that directory is ignored.
const TOOL_RESULT_REF_RE =
  /(?:^|[\s"'=(])((?:[^\s"'()]+\/)?tool-results\/[^\s"'()]+\.txt)\b/g;

function parseBooleanEnabled(raw: string | undefined): boolean {
  if (raw === undefined) return true;
  return !new Set(['', '0', 'false', 'no', 'off']).has(
    raw.trim().toLowerCase(),
  );
}

function parseNonNegativeInt(
  raw: string | undefined,
  fallback: number,
): number {
  if (raw === undefined || raw.trim() === '') return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n >= 0 ? Math.floor(n) : fallback;
}

export function resolveSessionArtifactRetentionConfig(
  env: NodeJS.ProcessEnv = process.env,
): SessionArtifactRetentionConfig {
  return {
    enabled: parseBooleanEnabled(env.SESSION_ARTIFACT_RETENTION),
    maxToolResultBytes: parseNonNegativeInt(
      env.SESSION_TOOL_RESULT_MAX_BYTES,
      DEFAULT_MAX_TOOL_RESULT_BYTES,
    ),
    minToolResultAgeMs: parseNonNegativeInt(
      env.SESSION_TOOL_RESULT_MIN_AGE_MS,
      DEFAULT_MIN_TOOL_RESULT_AGE_MS,
    ),
    keepRecentImages: parseNonNegativeInt(
      env.SESSION_INLINE_IMAGE_KEEP_RECENT,
      DEFAULT_KEEP_RECENT_IMAGES,
    ),
  };
}

function emptyResult(
  transcriptPath: string | null,
): SessionArtifactRetentionResult {
  return {
    transcriptPath,
    rewritten: false,
    imageBlocksReplaced: 0,
    toolResultRefsReplaced: 0,
    toolResultFilesDeleted: 0,
  };
}

function findTranscriptPath(
  dataDir: string,
  groupFolder: string,
  sessionName: string,
  sessionId: string,
): string | null {
  const projectsDir = path.join(
    dataDir,
    'sessions',
    groupFolder,
    sessionName,
    '.claude',
    'projects',
  );
  const fast = path.join(projectsDir, '-workspace-group', `${sessionId}.jsonl`);
  if (fs.existsSync(fast)) return fast;

  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(projectsDir, { withFileTypes: true });
  } catch {
    return null;
  }
  for (const entry of entries) {
    if (!entry.isDirectory() || entry.isSymbolicLink()) continue;
    const candidate = path.join(projectsDir, entry.name, `${sessionId}.jsonl`);
    if (fs.existsSync(candidate)) return candidate;
  }
  return null;
}

function isImageBlock(value: unknown): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const block = value as Record<string, unknown>;
  return block.type === 'image';
}

function replaceImageBlock(): Record<string, string> {
  return {
    type: 'text',
    text: '[screenshot evicted from resumed session context — re-run the browser screenshot if current visual state is needed]',
  };
}

function countImageBlocks(value: unknown): number {
  if (Array.isArray(value))
    return value.reduce((n, v) => n + countImageBlocks(v), 0);
  if (!value || typeof value !== 'object') return 0;
  let count = isImageBlock(value) ? 1 : 0;
  for (const child of Object.values(value as Record<string, unknown>)) {
    count += countImageBlocks(child);
  }
  return count;
}

function replaceOldImageBlocks(
  value: unknown,
  keep: { remaining: number },
): number {
  if (Array.isArray(value)) {
    let count = 0;
    for (let i = value.length - 1; i >= 0; i--) {
      const child = value[i];
      if (isImageBlock(child)) {
        if (keep.remaining > 0) {
          keep.remaining--;
        } else {
          value[i] = replaceImageBlock();
          count++;
        }
        continue;
      }
      count += replaceOldImageBlocks(child, keep);
    }
    return count;
  }
  if (!value || typeof value !== 'object') return 0;
  const obj = value as Record<string, unknown>;
  let count = 0;
  // Object key order is structural, not temporal. Recency is only meaningful
  // across JSONL lines and ordered content arrays, so object children are
  // traversed in ordinary insertion order while arrays are walked newest-first.
  for (const [key, child] of Object.entries(obj)) {
    if (isImageBlock(child)) {
      if (keep.remaining > 0) {
        keep.remaining--;
      } else {
        obj[key] = replaceImageBlock();
        count++;
      }
      continue;
    }
    count += replaceOldImageBlocks(child, keep);
  }
  return count;
}

function resolveToolResultRef(
  projectDir: string,
  rawRef: string,
): string | null {
  const normalized = rawRef.replace(/\\/g, '/');
  const idx = normalized.lastIndexOf('tool-results/');
  if (idx < 0) return null;
  const rel = normalized.slice(idx);
  const resolved = path.resolve(projectDir, rel);
  const toolResultsDir = path.resolve(projectDir, 'tool-results');
  if (
    resolved !== toolResultsDir &&
    !resolved.startsWith(toolResultsDir + path.sep)
  ) {
    return null;
  }
  return resolved;
}

function shouldPruneFile(
  filePath: string,
  config: SessionArtifactRetentionConfig,
  nowMs: number,
): boolean {
  let stat: fs.Stats;
  try {
    stat = fs.lstatSync(filePath);
  } catch {
    return false;
  }
  if (!stat.isFile() || stat.isSymbolicLink()) return false;
  if (stat.size <= config.maxToolResultBytes) return false;
  if (nowMs - stat.mtimeMs < config.minToolResultAgeMs) return false;
  return true;
}

function makeToolResultStub(filePath: string, size: number): string {
  return `[large tool-result side-file evicted from resumed session context: ${path.basename(filePath)} (${size} bytes) — re-run the tool if exact prior output is needed]`;
}

function mutateStringRefs(
  value: string,
  projectDir: string,
  config: SessionArtifactRetentionConfig,
  nowMs: number,
  filesToDelete: Map<string, number>,
): { value: string; replaced: number } {
  let replaced = 0;
  const next = value.replace(TOOL_RESULT_REF_RE, (match, ref: string) => {
    const prefixLen = match.length - ref.length;
    const prefix = match.slice(0, prefixLen);
    const filePath = resolveToolResultRef(projectDir, ref);
    if (!filePath || !shouldPruneFile(filePath, config, nowMs)) return match;

    const size = fs.lstatSync(filePath).size;
    filesToDelete.set(filePath, size);
    replaced++;
    return prefix + makeToolResultStub(filePath, size);
  });
  return { value: next, replaced };
}

function replaceToolResultRefs(
  value: unknown,
  projectDir: string,
  config: SessionArtifactRetentionConfig,
  nowMs: number,
  filesToDelete: Map<string, number>,
): { replaced: number } {
  if (typeof value === 'string') {
    return mutateStringRefs(value, projectDir, config, nowMs, filesToDelete);
  }
  if (Array.isArray(value)) {
    let replaced = 0;
    for (let i = 0; i < value.length; i++) {
      if (typeof value[i] === 'string') {
        const out = mutateStringRefs(
          value[i],
          projectDir,
          config,
          nowMs,
          filesToDelete,
        );
        value[i] = out.value;
        replaced += out.replaced;
      } else {
        const out = replaceToolResultRefs(
          value[i],
          projectDir,
          config,
          nowMs,
          filesToDelete,
        );
        replaced += out.replaced;
      }
    }
    return { replaced };
  }
  if (!value || typeof value !== 'object') return { replaced: 0 };

  let replaced = 0;
  const obj = value as Record<string, unknown>;
  for (const [key, child] of Object.entries(obj)) {
    if (typeof child === 'string') {
      const out = mutateStringRefs(
        child,
        projectDir,
        config,
        nowMs,
        filesToDelete,
      );
      obj[key] = out.value;
      replaced += out.replaced;
    } else {
      const out = replaceToolResultRefs(
        child,
        projectDir,
        config,
        nowMs,
        filesToDelete,
      );
      replaced += out.replaced;
    }
  }
  return { replaced };
}

export function pruneSessionArtifacts(options: {
  dataDir: string;
  groupFolder: string;
  sessionName: string;
  sessionId: string;
  config?: SessionArtifactRetentionConfig;
  nowMs?: number;
}): SessionArtifactRetentionResult {
  const config = options.config ?? resolveSessionArtifactRetentionConfig();
  const transcriptPath = findTranscriptPath(
    options.dataDir,
    options.groupFolder,
    options.sessionName,
    options.sessionId,
  );
  const result = emptyResult(transcriptPath);
  if (!config.enabled || !transcriptPath) return result;

  let raw: string;
  try {
    raw = fs.readFileSync(transcriptPath, 'utf8');
  } catch {
    return result;
  }
  if (!raw.trim()) return result;

  const lines = raw.split(/\n/);
  const parsedLines: Array<unknown | null> = [];
  let totalImages = 0;
  for (const [lineIndex, line] of lines.entries()) {
    if (!line.trim()) {
      parsedLines.push(null);
      continue;
    }
    try {
      const parsed = JSON.parse(line);
      totalImages += countImageBlocks(parsed);
      parsedLines.push(parsed);
    } catch (err) {
      logger.warn(
        { transcriptPath, lineIndex, err },
        'session_artifact_retention_parse_failed',
      );
      return result;
    }
  }

  let imagesToKeep = Math.min(config.keepRecentImages, totalImages);
  const projectDir = path.dirname(transcriptPath);
  const filesToDelete = new Map<string, number>();
  const nowMs = options.nowMs ?? Date.now();

  for (let i = parsedLines.length - 1; i >= 0; i--) {
    const parsed = parsedLines[i];
    if (parsed === null) continue;
    result.imageBlocksReplaced += replaceOldImageBlocks(parsed, {
      remaining: imagesToKeep,
    });
    // Re-count after mutation: replaced blocks are now text stubs, so this
    // subtracts only images retained in this line from the cross-line budget.
    imagesToKeep = Math.max(0, imagesToKeep - countImageBlocks(parsed));

    const refs = replaceToolResultRefs(
      parsed,
      projectDir,
      config,
      nowMs,
      filesToDelete,
    );
    result.toolResultRefsReplaced += refs.replaced;
  }

  if (result.imageBlocksReplaced === 0 && result.toolResultRefsReplaced === 0) {
    logger.debug(
      {
        transcriptPath,
        groupFolder: options.groupFolder,
        sessionName: options.sessionName,
      },
      'session_artifact_retention_noop',
    );
    return result;
  }

  const rewritten = parsedLines
    .map((parsed, i) => (parsed === null ? lines[i] : JSON.stringify(parsed)))
    .join('\n');
  const tmp = `${transcriptPath}.tmp.${process.pid}`;
  try {
    fs.writeFileSync(tmp, rewritten, 'utf8');
    fs.renameSync(tmp, transcriptPath);
  } catch {
    try {
      fs.rmSync(tmp, { force: true });
    } catch {
      // best effort cleanup only
    }
    return emptyResult(transcriptPath);
  }

  for (const filePath of filesToDelete.keys()) {
    try {
      fs.unlinkSync(filePath);
      result.toolResultFilesDeleted++;
    } catch {
      // The transcript now contains a stub, so a failed unlink is only a
      // disk-space concern; never fail the container spawn because of it.
    }
  }
  result.rewritten = true;
  return result;
}
