import { describe, expect, it, vi } from 'vitest';

import { applyPromptCacheTtl } from './prompt-cache-ttl.js';
import type { ContainerContext } from './usage-log.js';

const mainCtx: ContainerContext = {
  group: 'telegram_swarm',
  tier: 'main',
  session: 'default',
  task_id: null,
  message_id: '123',
};

function buf(value: unknown): Buffer {
  return Buffer.from(JSON.stringify(value), 'utf8');
}

describe('applyPromptCacheTtl', () => {
  it('upgrades existing ephemeral cache controls to 1h for telegram_swarm default session', () => {
    const body = buf({
      system: [
        { type: 'text', text: 'frozen', cache_control: { type: 'ephemeral' } },
        { type: 'text', text: 'volatile' },
      ],
      tools: [
        {
          name: 'Bash',
          description: 'tool',
          cache_control: { type: 'ephemeral', ttl: '5m' },
        },
      ],
    });

    const result = applyPromptCacheTtl(
      '/v1/messages?beta=true',
      'POST',
      body,
      mainCtx,
      {},
    );

    expect(result.applied).toBe(true);
    expect(result.stats.ttlApplied).toBe(2);
    const parsed = JSON.parse(result.body.toString('utf8'));
    expect(parsed.system[0].cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
    expect(parsed.tools[0].cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
  });

  it('upgrades message content block cache controls', () => {
    const body = buf({
      messages: [
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: 'large cached content block',
              cache_control: { type: 'ephemeral' },
            },
          ],
        },
      ],
    });

    const result = applyPromptCacheTtl(
      '/v1/messages',
      'POST',
      body,
      mainCtx,
      {},
    );

    expect(result.applied).toBe(true);
    expect(result.stats.ttlApplied).toBe(1);
    const parsed = JSON.parse(result.body.toString('utf8'));
    expect(parsed.messages[0].content[0].cache_control).toEqual({
      type: 'ephemeral',
      ttl: '1h',
    });
  });

  it('does not rewrite cache_control fields outside known Anthropic request locations', () => {
    const body = buf({
      metadata: { cache_control: { type: 'ephemeral' } },
      tools: [
        {
          name: 'Echo',
          input_schema: {
            type: 'object',
            properties: {
              payload: { cache_control: { type: 'ephemeral' } },
            },
          },
        },
      ],
    });

    const result = applyPromptCacheTtl(
      '/v1/messages',
      'POST',
      body,
      mainCtx,
      {},
    );

    expect(result.applied).toBe(false);
    expect(result.body).toBe(body);
  });

  it('does not invent cache controls when the SDK emitted no breakpoint', () => {
    const body = buf({ system: [{ type: 'text', text: 'plain' }] });
    const result = applyPromptCacheTtl(
      '/v1/messages',
      'POST',
      body,
      mainCtx,
      {},
    );
    expect(result.applied).toBe(false);
    expect(result.body).toBe(body);
  });

  it('is scoped to configured main default-session groups', () => {
    const body = buf({
      system: [
        { type: 'text', text: 'x', cache_control: { type: 'ephemeral' } },
      ],
    });
    const trusted = { ...mainCtx, tier: 'trusted' as const };
    const maintenance = { ...mainCtx, session: 'maintenance' };
    const task = { ...mainCtx, task_id: 'heartbeat-1' };
    const otherGroup = { ...mainCtx, group: 'telegram_other' };

    expect(
      applyPromptCacheTtl('/v1/messages', 'POST', body, trusted, {}).applied,
    ).toBe(false);
    expect(
      applyPromptCacheTtl('/v1/messages', 'POST', body, maintenance, {})
        .applied,
    ).toBe(false);
    expect(
      applyPromptCacheTtl('/v1/messages', 'POST', body, task, {}).applied,
    ).toBe(false);
    expect(
      applyPromptCacheTtl('/v1/messages', 'POST', body, otherGroup, {}).applied,
    ).toBe(false);
  });

  it('allows explicit group rollout and global disable via env', () => {
    const body = buf({
      system: [
        { type: 'text', text: 'x', cache_control: { type: 'ephemeral' } },
      ],
    });
    const otherGroup = { ...mainCtx, group: 'signal_main' };

    expect(
      applyPromptCacheTtl('/v1/messages', 'POST', body, otherGroup, {
        NANOCLAW_PROMPT_CACHE_1H_GROUPS: 'signal_main',
      }).applied,
    ).toBe(true);
    for (const disabled of ['0', 'false', 'no', 'off', '']) {
      expect(
        applyPromptCacheTtl('/v1/messages', 'POST', body, mainCtx, {
          NANOCLAW_PROMPT_CACHE_1H: disabled,
        }).applied,
      ).toBe(false);
    }
  });

  it('passes malformed JSON through and reports SyntaxError only', () => {
    const onParseError = vi.fn();
    const body = Buffer.from('not-json{', 'utf8');
    const result = applyPromptCacheTtl(
      '/v1/messages',
      'POST',
      body,
      mainCtx,
      {},
      onParseError,
    );
    expect(result.applied).toBe(false);
    expect(result.body).toBe(body);
    expect(onParseError).toHaveBeenCalledOnce();
  });
});
