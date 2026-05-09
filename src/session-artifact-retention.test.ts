import fs from 'fs';
import os from 'os';
import path from 'path';
import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  pruneSessionArtifacts,
  resolveSessionArtifactRetentionConfig,
  type SessionArtifactRetentionConfig,
} from './session-artifact-retention.js';
import { logger } from './logger.js';

const tmpDirs: string[] = [];

function makeTmp(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'nanoclaw-retention-'));
  tmpDirs.push(dir);
  return dir;
}

afterEach(() => {
  vi.restoreAllMocks();
  for (const dir of tmpDirs.splice(0)) {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

const baseConfig: SessionArtifactRetentionConfig = {
  enabled: true,
  maxToolResultBytes: 10,
  minToolResultAgeMs: 1_000,
  keepRecentImages: 1,
};

function writeSession(
  dataDir: string,
  lines: unknown[],
  sessionId = 'sid1',
  projectSlug = '-workspace-group',
) {
  const projectDir = path.join(
    dataDir,
    'sessions',
    'telegram_main',
    'default',
    '.claude',
    'projects',
    projectSlug,
  );
  fs.mkdirSync(path.join(projectDir, 'tool-results'), { recursive: true });
  const transcriptPath = path.join(projectDir, `${sessionId}.jsonl`);
  fs.writeFileSync(
    transcriptPath,
    lines.map((line) => JSON.stringify(line)).join('\n') + '\n',
    'utf8',
  );
  return { projectDir, transcriptPath };
}

describe('session artifact retention', () => {
  it('replaces older inline screenshot image blocks while keeping the most recent images', () => {
    const dataDir = makeTmp();
    const { transcriptPath } = writeSession(dataDir, [
      {
        type: 'user',
        message: {
          content: [
            {
              type: 'tool_result',
              content: [
                { type: 'image', source: { type: 'base64', data: 'old' } },
              ],
            },
          ],
        },
      },
      {
        type: 'user',
        message: {
          content: [
            {
              type: 'tool_result',
              content: [
                { type: 'image', source: { type: 'base64', data: 'new' } },
              ],
            },
          ],
        },
      },
    ]);

    const result = pruneSessionArtifacts({
      dataDir,
      groupFolder: 'telegram_main',
      sessionName: 'default',
      sessionId: 'sid1',
      config: baseConfig,
    });

    expect(result.rewritten).toBe(true);
    expect(result.imageBlocksReplaced).toBe(1);
    const [first, second] = fs
      .readFileSync(transcriptPath, 'utf8')
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line));
    expect(first.message.content[0].content[0]).toMatchObject({
      type: 'text',
    });
    expect(second.message.content[0].content[0]).toMatchObject({
      type: 'image',
      source: { data: 'new' },
    });
  });

  it('evicts all inline screenshot image blocks when keepRecentImages is 0', () => {
    const dataDir = makeTmp();
    const { transcriptPath } = writeSession(dataDir, [
      {
        type: 'user',
        message: {
          content: [
            {
              type: 'tool_result',
              content: [
                { type: 'image', source: { type: 'base64', data: 'only' } },
              ],
            },
          ],
        },
      },
    ]);

    const result = pruneSessionArtifacts({
      dataDir,
      groupFolder: 'telegram_main',
      sessionName: 'default',
      sessionId: 'sid1',
      config: { ...baseConfig, keepRecentImages: 0 },
    });

    expect(result.rewritten).toBe(true);
    expect(result.imageBlocksReplaced).toBe(1);
    const parsed = JSON.parse(fs.readFileSync(transcriptPath, 'utf8').trim());
    expect(parsed.message.content[0].content[0]).toMatchObject({
      type: 'text',
    });
  });

  it('does not treat non-image base64 document blocks as screenshots', () => {
    const dataDir = makeTmp();
    const { transcriptPath } = writeSession(dataDir, [
      {
        type: 'user',
        message: {
          content: [
            {
              type: 'document',
              source: {
                type: 'base64',
                media_type: 'application/pdf',
                data: 'pdf',
              },
            },
          ],
        },
      },
    ]);

    const result = pruneSessionArtifacts({
      dataDir,
      groupFolder: 'telegram_main',
      sessionName: 'default',
      sessionId: 'sid1',
      config: { ...baseConfig, keepRecentImages: 0 },
    });

    expect(result.rewritten).toBe(false);
    expect(fs.readFileSync(transcriptPath, 'utf8')).toContain(
      'application/pdf',
    );
  });

  it('rewrites large old tool-result side-file references and deletes the side-file', () => {
    const dataDir = makeTmp();
    const { projectDir, transcriptPath } = writeSession(dataDir, [
      {
        type: 'user',
        message: {
          content: [
            {
              type: 'tool_result',
              content: 'see tool-results/mcp-superhuman-get_thread-1.txt',
            },
          ],
        },
      },
    ]);
    const sideFile = path.join(
      projectDir,
      'tool-results',
      'mcp-superhuman-get_thread-1.txt',
    );
    fs.writeFileSync(sideFile, 'x'.repeat(100), 'utf8');
    const old = new Date(Date.now() - 10_000);
    fs.utimesSync(sideFile, old, old);

    const result = pruneSessionArtifacts({
      dataDir,
      groupFolder: 'telegram_main',
      sessionName: 'default',
      sessionId: 'sid1',
      config: baseConfig,
      nowMs: Date.now(),
    });

    expect(result.rewritten).toBe(true);
    expect(result.toolResultRefsReplaced).toBe(1);
    expect(result.toolResultFilesDeleted).toBe(1);
    expect(fs.existsSync(sideFile)).toBe(false);
    expect(fs.readFileSync(transcriptPath, 'utf8')).toContain(
      'large tool-result side-file evicted',
    );
  });

  it('rejects symlinked tool-result side-files', () => {
    const dataDir = makeTmp();
    const { projectDir, transcriptPath } = writeSession(dataDir, [
      {
        type: 'user',
        message: {
          content: [{ type: 'tool_result', content: 'tool-results/link.txt' }],
        },
      },
    ]);
    const target = path.join(projectDir, 'target.txt');
    const link = path.join(projectDir, 'tool-results', 'link.txt');
    fs.writeFileSync(target, 'x'.repeat(100), 'utf8');
    fs.symlinkSync(target, link);

    const result = pruneSessionArtifacts({
      dataDir,
      groupFolder: 'telegram_main',
      sessionName: 'default',
      sessionId: 'sid1',
      config: baseConfig,
      nowMs: Date.now() + 10_000,
    });

    expect(result.rewritten).toBe(false);
    expect(fs.lstatSync(link).isSymbolicLink()).toBe(true);
    expect(fs.readFileSync(transcriptPath, 'utf8')).toContain(
      'tool-results/link.txt',
    );
  });

  it('rejects path traversal in tool-result references', () => {
    const dataDir = makeTmp();
    const { projectDir, transcriptPath } = writeSession(dataDir, [
      {
        type: 'user',
        message: {
          content: [
            {
              type: 'tool_result',
              content: 'tool-results/../../outside.txt',
            },
          ],
        },
      },
    ]);
    const outside = path.resolve(
      projectDir,
      'tool-results',
      '..',
      '..',
      'outside.txt',
    );
    fs.writeFileSync(outside, 'x'.repeat(100), 'utf8');
    const old = new Date(Date.now() - 10_000);
    fs.utimesSync(outside, old, old);

    const result = pruneSessionArtifacts({
      dataDir,
      groupFolder: 'telegram_main',
      sessionName: 'default',
      sessionId: 'sid1',
      config: baseConfig,
      nowMs: Date.now(),
    });

    expect(result.rewritten).toBe(false);
    expect(fs.existsSync(outside)).toBe(true);
    expect(fs.readFileSync(transcriptPath, 'utf8')).toContain(
      'tool-results/../../outside.txt',
    );
  });

  it('finds transcripts under non-canonical project slug directories', () => {
    const dataDir = makeTmp();
    const { transcriptPath } = writeSession(
      dataDir,
      [
        {
          type: 'user',
          message: {
            content: [
              {
                type: 'tool_result',
                content: [
                  { type: 'image', source: { type: 'base64', data: 'old' } },
                ],
              },
            ],
          },
        },
      ],
      'sid1',
      'custom-project-slug',
    );

    const result = pruneSessionArtifacts({
      dataDir,
      groupFolder: 'telegram_main',
      sessionName: 'default',
      sessionId: 'sid1',
      config: { ...baseConfig, keepRecentImages: 0 },
    });

    expect(result.rewritten).toBe(true);
    expect(result.transcriptPath).toBe(transcriptPath);
  });

  it('logs and aborts without rewriting when a JSONL line is corrupt', () => {
    const dataDir = makeTmp();
    const { transcriptPath } = writeSession(dataDir, [
      {
        type: 'user',
        message: {
          content: [{ type: 'image', source: { type: 'base64', data: 'old' } }],
        },
      },
    ]);
    fs.appendFileSync(transcriptPath, '{not-json}\n', 'utf8');
    const warn = vi.spyOn(logger, 'warn').mockImplementation(() => undefined);

    const result = pruneSessionArtifacts({
      dataDir,
      groupFolder: 'telegram_main',
      sessionName: 'default',
      sessionId: 'sid1',
      config: { ...baseConfig, keepRecentImages: 0 },
    });

    expect(result.rewritten).toBe(false);
    expect(warn).toHaveBeenCalledWith(
      expect.objectContaining({ transcriptPath, lineIndex: 1 }),
      'session_artifact_retention_parse_failed',
    );
    expect(fs.readFileSync(transcriptPath, 'utf8')).toContain('{not-json}');
  });

  it('logs no-op runs at debug level', () => {
    const dataDir = makeTmp();
    writeSession(dataDir, [
      { type: 'user', message: { content: [{ type: 'text', text: 'hi' }] } },
    ]);
    const debug = vi.spyOn(logger, 'debug').mockImplementation(() => undefined);

    const result = pruneSessionArtifacts({
      dataDir,
      groupFolder: 'telegram_main',
      sessionName: 'default',
      sessionId: 'sid1',
      config: baseConfig,
    });

    expect(result.rewritten).toBe(false);
    expect(debug).toHaveBeenCalledWith(
      expect.objectContaining({
        groupFolder: 'telegram_main',
        sessionName: 'default',
      }),
      'session_artifact_retention_noop',
    );
  });

  it('keeps small or recent side-files untouched', () => {
    const dataDir = makeTmp();
    const { projectDir, transcriptPath } = writeSession(dataDir, [
      {
        type: 'user',
        message: {
          content: [
            { type: 'tool_result', content: 'tool-results/small.txt' },
            { type: 'tool_result', content: 'tool-results/recent.txt' },
          ],
        },
      },
    ]);
    const small = path.join(projectDir, 'tool-results', 'small.txt');
    const recent = path.join(projectDir, 'tool-results', 'recent.txt');
    fs.writeFileSync(small, 'tiny', 'utf8');
    fs.writeFileSync(recent, 'x'.repeat(100), 'utf8');

    const result = pruneSessionArtifacts({
      dataDir,
      groupFolder: 'telegram_main',
      sessionName: 'default',
      sessionId: 'sid1',
      config: baseConfig,
      nowMs: Date.now(),
    });

    expect(result.rewritten).toBe(false);
    expect(fs.existsSync(small)).toBe(true);
    expect(fs.existsSync(recent)).toBe(true);
    expect(fs.readFileSync(transcriptPath, 'utf8')).toContain(
      'tool-results/small.txt',
    );
  });

  it('supports common falsey env values for the retention feature flag', () => {
    for (const value of ['0', 'false', 'no', 'off', '']) {
      expect(
        resolveSessionArtifactRetentionConfig({
          SESSION_ARTIFACT_RETENTION: value,
        }).enabled,
      ).toBe(false);
    }
    expect(resolveSessionArtifactRetentionConfig({}).enabled).toBe(true);
  });
});
