import { describe, it, expect, vi } from 'vitest';

import {
  DEAD_TOOL_NAMES,
  TRIMMED_BASH_DESCRIPTION,
  applyWireToolFilter,
  filterToolsInBody,
  isInterceptorEnabled,
  isMessagesEndpoint,
} from './wire-tool-filter.js';

describe('wire-tool-filter / DEAD_TOOL_NAMES', () => {
  it('contains the 15 SDK-builtin tools approved for removal', () => {
    expect(DEAD_TOOL_NAMES.size).toBe(15);
    for (const name of [
      'NotebookEdit',
      'PushNotification',
      'EnterPlanMode',
      'ExitPlanMode',
      'EnterWorktree',
      'ExitWorktree',
      'RemoteTrigger',
      'ListMcpResourcesTool',
      'ReadMcpResourceTool',
      'CronCreate',
      'CronDelete',
      'CronList',
      'ScheduleWakeup',
      'Monitor',
      'AskUserQuestion',
    ]) {
      expect(DEAD_TOOL_NAMES.has(name)).toBe(true);
    }
  });

  it('does NOT contain explicitly kept tools (defense-in-depth)', () => {
    for (const name of [
      'TodoWrite',
      'TeamCreate',
      'TeamDelete',
      'SendMessage',
      'TaskOutput',
      'TaskStop',
      'Bash',
      'Read',
      'Write',
      'Edit',
      'Glob',
      'Grep',
      'WebFetch',
      'WebSearch',
      'Agent',
    ]) {
      expect(DEAD_TOOL_NAMES.has(name)).toBe(false);
    }
  });
});

describe('wire-tool-filter / isMessagesEndpoint', () => {
  it('matches /v1/messages with and without query string', () => {
    expect(isMessagesEndpoint('/v1/messages')).toBe(true);
    expect(isMessagesEndpoint('/v1/messages?beta=true')).toBe(true);
  });

  it('rejects unrelated paths', () => {
    expect(isMessagesEndpoint('/v1/complete')).toBe(false);
    expect(isMessagesEndpoint('/api/oauth/claude_cli/create_api_key')).toBe(
      false,
    );
    expect(isMessagesEndpoint(undefined)).toBe(false);
    expect(isMessagesEndpoint('')).toBe(false);
  });
});

describe('wire-tool-filter / isInterceptorEnabled', () => {
  it('defaults ON when STRIP_DEAD_TOOLS is unset', () => {
    expect(isInterceptorEnabled({})).toBe(true);
  });

  it('stays ON for STRIP_DEAD_TOOLS=1 or any non-"0" value', () => {
    expect(isInterceptorEnabled({ STRIP_DEAD_TOOLS: '1' })).toBe(true);
    expect(isInterceptorEnabled({ STRIP_DEAD_TOOLS: 'true' })).toBe(true);
  });

  it('disables only on STRIP_DEAD_TOOLS=0', () => {
    expect(isInterceptorEnabled({ STRIP_DEAD_TOOLS: '0' })).toBe(false);
  });
});

describe('wire-tool-filter / filterToolsInBody', () => {
  it('is a no-op when tools field is absent', () => {
    const body = { model: 'claude-opus-4-7', messages: [] };
    const stats = filterToolsInBody(body);
    expect(stats).toEqual({ toolsStripped: 0, descriptionsTrimmed: 0 });
    expect(body).toEqual({ model: 'claude-opus-4-7', messages: [] });
  });

  it('strips dead tools by name and preserves others', () => {
    const body = {
      tools: [
        { name: 'Bash', description: 'short' },
        { name: 'NotebookEdit', description: 'x' },
        { name: 'Monitor', description: 'x' },
        { name: 'TodoWrite', description: 'kept' },
        { name: 'AskUserQuestion', description: 'x' },
        { name: 'Read', description: 'r' },
      ],
    };
    const stats = filterToolsInBody(body);
    expect(stats.toolsStripped).toBe(3);
    expect(body.tools.map((t) => t.name)).toEqual([
      'Bash',
      'TodoWrite',
      'Read',
    ]);
  });

  it('trims Bash description in place', () => {
    const body = {
      tools: [
        {
          name: 'Bash',
          description: 'A very very long original Bash description...',
        },
      ],
    };
    const stats = filterToolsInBody(body);
    expect(stats.descriptionsTrimmed).toBe(1);
    expect(body.tools[0].description).toBe(TRIMMED_BASH_DESCRIPTION);
  });

  it('does not double-trim Bash description', () => {
    const body = {
      tools: [{ name: 'Bash', description: TRIMMED_BASH_DESCRIPTION }],
    };
    const stats = filterToolsInBody(body);
    expect(stats.descriptionsTrimmed).toBe(0);
  });

  it('handles empty tools array', () => {
    const body: { tools: { name?: unknown }[] } = { tools: [] };
    const stats = filterToolsInBody(body);
    expect(stats).toEqual({ toolsStripped: 0, descriptionsTrimmed: 0 });
    expect(body.tools).toEqual([]);
  });

  it('does not mutate non-array tools field', () => {
    const body = { tools: 'not-an-array' as unknown as never };
    const stats = filterToolsInBody(body);
    expect(stats).toEqual({ toolsStripped: 0, descriptionsTrimmed: 0 });
    expect(body.tools).toBe('not-an-array');
  });
});

describe('wire-tool-filter / applyWireToolFilter', () => {
  function buf(obj: unknown): Buffer {
    return Buffer.from(JSON.stringify(obj), 'utf8');
  }

  it('passthrough when STRIP_DEAD_TOOLS=0', () => {
    const original = buf({
      tools: [{ name: 'NotebookEdit', description: 'x' }],
    });
    const result = applyWireToolFilter('/v1/messages', 'POST', original, {
      STRIP_DEAD_TOOLS: '0',
    });
    expect(result.applied).toBe(false);
    expect(result.body).toBe(original);
    expect(result.stats).toEqual({ toolsStripped: 0, descriptionsTrimmed: 0 });
  });

  it('passthrough for non-messages endpoints', () => {
    const original = buf({
      tools: [{ name: 'NotebookEdit', description: 'x' }],
    });
    const result = applyWireToolFilter(
      '/api/oauth/claude_cli/create_api_key',
      'POST',
      original,
      {},
    );
    expect(result.applied).toBe(false);
    expect(result.body).toBe(original);
  });

  it('passthrough for non-POST requests', () => {
    const original = buf({
      tools: [{ name: 'NotebookEdit', description: 'x' }],
    });
    const result = applyWireToolFilter('/v1/messages', 'GET', original, {});
    expect(result.applied).toBe(false);
    expect(result.body).toBe(original);
  });

  it('passthrough for empty body', () => {
    const result = applyWireToolFilter(
      '/v1/messages',
      'POST',
      Buffer.alloc(0),
      {},
    );
    expect(result.applied).toBe(false);
    expect(result.body.length).toBe(0);
  });

  it('passthrough + log on invalid JSON body', () => {
    const original = Buffer.from('not-json{', 'utf8');
    const onParseError = vi.fn();
    const result = applyWireToolFilter(
      '/v1/messages',
      'POST',
      original,
      {},
      onParseError,
    );
    expect(result.applied).toBe(false);
    expect(result.body).toBe(original);
    expect(onParseError).toHaveBeenCalledOnce();
  });

  it('rethrows non-SyntaxError exceptions (programming bugs surface loudly)', () => {
    // Hand `applyWireToolFilter` a Buffer whose `toString` throws a
    // non-SyntaxError (simulating a TypeError from a future refactor).
    // The narrowed catch must rethrow rather than swallow per
    // `error-handling.Specific Exceptions`.
    const sabotaged = {
      length: 4,
      toString() {
        throw new TypeError('simulated programming bug');
      },
    } as unknown as Buffer;
    const onParseError = vi.fn();
    expect(() =>
      applyWireToolFilter('/v1/messages', 'POST', sabotaged, {}, onParseError),
    ).toThrow(TypeError);
    expect(onParseError).not.toHaveBeenCalled();
  });

  it('passthrough when tools field is absent (non-tool-use messages)', () => {
    const original = buf({ model: 'claude-opus-4-7', messages: [] });
    const result = applyWireToolFilter('/v1/messages', 'POST', original, {});
    expect(result.applied).toBe(false);
    expect(result.body).toBe(original);
  });

  it('strips dead tools and trims Bash on /v1/messages POST (default ON)', () => {
    const original = buf({
      model: 'claude-opus-4-7',
      tools: [
        { name: 'Bash', description: 'long original' },
        { name: 'NotebookEdit', description: 'x' },
        { name: 'EnterPlanMode', description: 'x' },
        { name: 'TodoWrite', description: 'kept' },
        { name: 'Monitor', description: 'x' },
      ],
    });

    const result = applyWireToolFilter('/v1/messages', 'POST', original, {});

    expect(result.applied).toBe(true);
    expect(result.stats.toolsStripped).toBe(3);
    expect(result.stats.descriptionsTrimmed).toBe(1);

    const parsed = JSON.parse(result.body.toString('utf8'));
    expect(parsed.tools.map((t: { name: string }) => t.name)).toEqual([
      'Bash',
      'TodoWrite',
    ]);
    expect(parsed.tools[0].description).toBe(TRIMMED_BASH_DESCRIPTION);
  });

  it('handles /v1/messages?beta=true with query string', () => {
    const original = buf({
      tools: [{ name: 'NotebookEdit', description: 'x' }],
    });
    const result = applyWireToolFilter(
      '/v1/messages?beta=true',
      'POST',
      original,
      {},
    );
    expect(result.applied).toBe(true);
    const parsed = JSON.parse(result.body.toString('utf8'));
    expect(parsed.tools).toEqual([]);
  });

  it('handles tools[] containing only dead tools (empty result)', () => {
    const original = buf({
      tools: [
        { name: 'NotebookEdit', description: 'x' },
        { name: 'Monitor', description: 'x' },
      ],
    });
    const result = applyWireToolFilter('/v1/messages', 'POST', original, {});
    expect(result.applied).toBe(true);
    expect(result.stats.toolsStripped).toBe(2);
    const parsed = JSON.parse(result.body.toString('utf8'));
    expect(parsed.tools).toEqual([]);
  });
});
