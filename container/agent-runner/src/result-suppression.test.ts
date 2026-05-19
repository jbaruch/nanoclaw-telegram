import { describe, it, expect } from 'vitest';
import { buildSuccessOutput } from './result-suppression.js';

describe('buildSuccessOutput', () => {
  it('populates result and chat_displayed when send tool succeeded and text exists (#581 wrapper-skill case)', () => {
    // The wrapper case: agent finished by calling send_message AND the
    // SDK still emitted a closing-thought textResult. Result must still
    // be carried for task_run_logs.result; chat_displayed flags the
    // orchestrator + task-scheduler to skip their chat-echo path.
    const out = buildSuccessOutput(
      'closing thought from agent',
      true,
      'sess-1',
      { input_tokens: 10, output_tokens: 20 },
    );
    expect(out).toEqual({
      status: 'success',
      result: 'closing thought from agent',
      newSessionId: 'sess-1',
      usage: { input_tokens: 10, output_tokens: 20 },
      chat_displayed: true,
    });
  });

  it('omits chat_displayed when send tool was not used (chat-echo path is the source of the reply)', () => {
    const out = buildSuccessOutput('reply text', false, 'sess-2', undefined);
    expect(out).toEqual({
      status: 'success',
      result: 'reply text',
      newSessionId: 'sess-2',
      usage: undefined,
    });
    expect(out).not.toHaveProperty('chat_displayed');
  });

  it('omits chat_displayed when no text result is present even though send succeeded', () => {
    // No textResult to echo, so chat-echo gating is moot; do not emit a
    // chat_displayed flag that downstream consumers might misread.
    const out = buildSuccessOutput(null, true, 'sess-3', undefined);
    expect(out).toEqual({
      status: 'success',
      result: null,
      newSessionId: 'sess-3',
      usage: undefined,
    });
    expect(out).not.toHaveProperty('chat_displayed');
  });

  it('treats empty string textResult as no-text (same shape as null)', () => {
    const out = buildSuccessOutput('', true, 'sess-4', undefined);
    expect(out.result).toBeNull();
    expect(out).not.toHaveProperty('chat_displayed');
  });

  it('preserves newSessionId and usage even when chat_displayed fires', () => {
    const usage = {
      input_tokens: 100,
      output_tokens: 200,
      cache_read_input_tokens: 50,
    };
    const out = buildSuccessOutput('final text', true, 'sess-5', usage);
    expect(out.newSessionId).toBe('sess-5');
    expect(out.usage).toBe(usage);
    expect(out.chat_displayed).toBe(true);
  });
});
