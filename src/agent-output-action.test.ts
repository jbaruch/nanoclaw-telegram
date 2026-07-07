import { describe, it, expect } from 'vitest';
import {
  claimReplyAnchor,
  decideAgentOutputAction,
  stripInternalBlocks,
} from './agent-output-action.js';

describe('claimReplyAnchor (#722)', () => {
  it('claims a free anchor (previous turn already replied)', () => {
    const anchors: Record<string, string | undefined> = {};
    expect(claimReplyAnchor(anchors, 'chat@g.us', 'piped-1')).toBe(true);
    expect(anchors['chat@g.us']).toBe('piped-1');
  });

  it('refuses to clobber an unconsumed anchor (in-flight turn keeps its quote)', () => {
    // The #722 live repro: trigger sets the anchor, the turn runs long,
    // a piped batch arrives BEFORE the first reply — the final response
    // must still quote the trigger, not the pipe.
    const anchors: Record<string, string | undefined> = {
      'chat@g.us': 'trigger-msg',
    };
    expect(claimReplyAnchor(anchors, 'chat@g.us', 'piped-1')).toBe(false);
    expect(anchors['chat@g.us']).toBe('trigger-msg');
  });

  it('full turn sequence: trigger held → pipe refused → consume → next pipe claims', () => {
    const anchors: Record<string, string | undefined> = {};
    // Turn start (unconditional assignment in processGroupMessages).
    anchors['chat@g.us'] = 'trigger-msg';
    // Mid-turn pipe: refused, in-flight quote preserved.
    expect(claimReplyAnchor(anchors, 'chat@g.us', 'piped-1')).toBe(false);
    // Output callback sends the reply and consumes the anchor.
    anchors['chat@g.us'] = undefined;
    // Next pipe is the next turn's trigger: claim lands.
    expect(claimReplyAnchor(anchors, 'chat@g.us', 'piped-2')).toBe(true);
    expect(anchors['chat@g.us']).toBe('piped-2');
  });

  it('is per-chat: one chat holding its anchor does not block another', () => {
    const anchors: Record<string, string | undefined> = {
      'a@g.us': 'held',
    };
    expect(claimReplyAnchor(anchors, 'b@g.us', 'msg-b')).toBe(true);
    expect(anchors['a@g.us']).toBe('held');
    expect(anchors['b@g.us']).toBe('msg-b');
  });
});

describe('decideAgentOutputAction', () => {
  it('returns mark-displayed when chat_displayed=true and text is non-empty (#581 wrapper-skill case)', () => {
    // Wrapper case: agent finished via send_message AND the SDK still
    // emitted a closing-thought textResult. The agent-runner sets
    // chat_displayed=true. Caller must skip sendMessage + storeMessage
    // but still log the text + consume pendingReplyTo so a follow-up
    // doesn't reply to a stale message id.
    const action = decideAgentOutputAction({
      result: 'closing thought from agent',
      chat_displayed: true,
    });
    expect(action).toEqual({
      kind: 'mark-displayed',
      textForLog: 'closing thought from agent',
      rawLength: 'closing thought from agent'.length,
    });
  });

  it('returns send when chat_displayed is undefined and text is non-empty (#581 default path)', () => {
    // Default case: agent has no send_message call, the SDK textResult
    // IS the reply. Caller takes the chat-echo path.
    const action = decideAgentOutputAction({
      result: 'a normal reply',
    });
    expect(action).toEqual({
      kind: 'send',
      textForLog: 'a normal reply',
      rawLength: 'a normal reply'.length,
    });
  });

  it('returns reset-only when text is empty after stripping <internal> blocks', () => {
    // The agent is doing work (tool calls, thinking) but not producing
    // user-visible output. Caller resets the idle timer but skips the
    // send + storeMessage path. rawLength is reported so the caller
    // can still emit its "Agent output: N chars" telemetry log.
    const internalOnly = '<internal>thinking about the answer</internal>';
    const action = decideAgentOutputAction({
      result: internalOnly,
    });
    expect(action).toEqual({
      kind: 'reset-only',
      rawLength: internalOnly.length,
    });
  });

  it('returns reset-only when text is empty after stripping even if chat_displayed is set', () => {
    // chat_displayed without a textResult is effectively informational
    // (see ContainerOutput.chat_displayed doc); the empty-text branch
    // takes precedence so the caller still resets the idle timer and
    // skips the chat-echo path on the same reasoning as the no-flag
    // case.
    const action = decideAgentOutputAction({
      result: '<internal>private</internal>',
      chat_displayed: true,
    });
    expect(action.kind).toBe('reset-only');
  });

  it('returns noop when result is null (session-update markers)', () => {
    // The SDK emits non-result envelopes during a turn (tool calls,
    // session-update markers). `result: null` is the signal the caller
    // should ignore the event entirely — no log, no idle reset, no
    // send. Mirrors the outer `if (result.result)` gate in
    // processGroupMessages.
    expect(decideAgentOutputAction({ result: null })).toEqual({ kind: 'noop' });
    expect(decideAgentOutputAction({ result: undefined })).toEqual({
      kind: 'noop',
    });
  });

  it('returns noop when the streamedOutput envelope itself is null/undefined', () => {
    // Defensive: callers occasionally see undefined streamed events
    // from the SDK iterator. Helper must not throw.
    expect(decideAgentOutputAction(null)).toEqual({ kind: 'noop' });
    expect(decideAgentOutputAction(undefined)).toEqual({ kind: 'noop' });
  });

  it('treats empty-string result as noop (no text to dispatch)', () => {
    // Empty-string result is the same shape as null for routing
    // purposes — outer gate skips it.
    expect(decideAgentOutputAction({ result: '' })).toEqual({ kind: 'noop' });
  });

  it('keeps user-visible text when <internal> sits alongside real content', () => {
    // The strip pass removes internal blocks but the surrounding
    // user-visible text must survive intact. Caller's textForLog +
    // chat-echo body need the stripped text, not the raw text.
    const raw =
      'Here is your answer.<internal>self-critique notes</internal> Done.';
    const action = decideAgentOutputAction({
      result: raw,
    });
    expect(action).toEqual({
      kind: 'send',
      textForLog: 'Here is your answer. Done.',
      rawLength: raw.length,
    });
  });

  it('routes to mark-displayed when chat_displayed=true even with <internal> blocks present', () => {
    // The strip pass runs BEFORE the chat_displayed branch, so a
    // wrapper-skill closing thought wrapped in surrounding internal
    // notes still routes to mark-displayed once the strip leaves
    // visible text.
    const raw =
      '<internal>plan</internal>Reply was sent via send_message.<internal>end</internal>';
    const action = decideAgentOutputAction({
      result: raw,
      chat_displayed: true,
    });
    expect(action).toEqual({
      kind: 'mark-displayed',
      textForLog: 'Reply was sent via send_message.',
      rawLength: raw.length,
    });
  });
});

describe('stripInternalBlocks', () => {
  it('removes multi-line <internal> blocks', () => {
    const raw = 'before<internal>\nline1\nline2\n</internal>after';
    expect(stripInternalBlocks(raw)).toBe('beforeafter');
  });

  it('removes multiple <internal> blocks in one string', () => {
    expect(
      stripInternalBlocks('a<internal>x</internal>b<internal>y</internal>c'),
    ).toBe('abc');
  });

  it('trims surrounding whitespace from the result', () => {
    expect(stripInternalBlocks('  hello  ')).toBe('hello');
  });

  it('returns empty string when input is entirely internal', () => {
    expect(stripInternalBlocks('<internal>only internal</internal>')).toBe('');
  });
});
