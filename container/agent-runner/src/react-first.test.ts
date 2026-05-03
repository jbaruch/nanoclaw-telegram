import { describe, it, expect } from 'vitest';

import {
  REACT_FIRST_DEFAULT_EMOJI,
  ReactIpcWriter,
  ReactToMessageIpcPayload,
  decideReactFirst,
  runReactFirstHook,
} from './react-first.js';

const baseGate = {
  isScheduledTask: false,
  isSubagent: false,
  prompt: 'hey TestAssistant, how are things?',
  assistantName: 'TestAssistant',
};

describe('decideReactFirst', () => {
  it('reacts on a normal user-facing inbound', () => {
    expect(decideReactFirst(baseGate)).toEqual({ react: true });
  });

  it('skips when the submission is from a sub-agent', () => {
    expect(decideReactFirst({ ...baseGate, isSubagent: true })).toEqual({
      react: false,
      skippedBy: 'subagent',
    });
  });

  it('skips on a scheduled task', () => {
    expect(
      decideReactFirst({ ...baseGate, isScheduledTask: true }),
    ).toEqual({ react: false, skippedBy: 'scheduled-task' });
  });

  it('skips when the prompt is wrapped as [SCHEDULED TASK]', () => {
    expect(
      decideReactFirst({
        ...baseGate,
        prompt: '[SCHEDULED TASK]\n\nScript output:\n{...}\n\nInstructions:\n...',
      }),
    ).toEqual({ react: false, skippedBy: 'scheduled-task-prompt-wrap' });
  });

  it('skips when assistantName is missing', () => {
    expect(
      decideReactFirst({ ...baseGate, assistantName: undefined }),
    ).toEqual({ react: false, skippedBy: 'no-assistant-name' });
  });

  it('skips when assistantName is empty', () => {
    expect(
      decideReactFirst({ ...baseGate, assistantName: '' }),
    ).toEqual({ react: false, skippedBy: 'no-assistant-name' });
  });

  it('subagent gate beats scheduled-task gate', () => {
    expect(
      decideReactFirst({
        ...baseGate,
        isSubagent: true,
        isScheduledTask: true,
      }),
    ).toEqual({ react: false, skippedBy: 'subagent' });
  });

  it('scheduled-task gate beats prompt-wrap gate', () => {
    expect(
      decideReactFirst({
        ...baseGate,
        isScheduledTask: true,
        prompt: '[SCHEDULED TASK] x',
      }),
    ).toEqual({ react: false, skippedBy: 'scheduled-task' });
  });

  // #289 — addressed-ness gate. The orchestrator resolves
  // `addressedToUs` from isMain / 1:1-DM / trigger-match /
  // reply-to-our-bot and pipes it through ContainerInput.
  it('skips when the orchestrator marks the inbound not-addressed', () => {
    expect(
      decideReactFirst({ ...baseGate, addressedToUs: false }),
    ).toEqual({ react: false, skippedBy: 'not-addressed' });
  });

  it('reacts when the orchestrator marks the inbound addressed', () => {
    expect(
      decideReactFirst({ ...baseGate, addressedToUs: true }),
    ).toEqual({ react: true });
  });

  it('treats undefined addressedToUs as "no signal" and falls through to react', () => {
    // Legacy entry points (or paths that don't compute the flag)
    // should not regress the historical default. Channel-routed
    // inbounds — the path that produced the original leak — always
    // set the flag explicitly.
    expect(
      decideReactFirst({ ...baseGate, addressedToUs: undefined }),
    ).toEqual({ react: true });
  });

  it('subagent gate beats not-addressed gate', () => {
    // Subagent skip is more semantically useful for triage than
    // addressed-ness when both apply.
    expect(
      decideReactFirst({
        ...baseGate,
        isSubagent: true,
        addressedToUs: false,
      }),
    ).toEqual({ react: false, skippedBy: 'subagent' });
  });
});

describe('REACT_FIRST_DEFAULT_EMOJI', () => {
  it('is the eye emoji', () => {
    expect(REACT_FIRST_DEFAULT_EMOJI).toBe('👀');
  });
});

describe('runReactFirstHook', () => {
  const baseHookInput = {
    isScheduledTask: false,
    isSubagent: false,
    prompt: '<message id="msg_42" sender="Baruch">help me</message>',
    assistantName: 'TestAssistant',
    chatJid: '120363042@g.us',
    groupFolder: 'main',
    sessionName: 'default',
  };
  const fixedNow = () => new Date('2026-04-27T18:00:00.000Z');

  function makeRecordingWriter(): {
    writer: ReactIpcWriter;
    payloads: ReactToMessageIpcPayload[];
  } {
    const payloads: ReactToMessageIpcPayload[] = [];
    return {
      writer: (p) => {
        payloads.push(p);
      },
      payloads,
    };
  }

  it('emits the default emoji on a normal user-facing inbound', () => {
    const { writer, payloads } = makeRecordingWriter();
    const result = runReactFirstHook(baseHookInput, writer, fixedNow);
    expect(result).toEqual({ kind: 'emitted', emoji: '👀' });
    expect(payloads).toHaveLength(1);
    expect(payloads[0]).toEqual({
      type: 'react_to_message',
      chatJid: '120363042@g.us',
      groupFolder: 'main',
      sessionName: 'default',
      emoji: '👀',
      timestamp: '2026-04-27T18:00:00.000Z',
    });
  });

  it('skips with `subagent` reason without invoking the writer', () => {
    const { writer, payloads } = makeRecordingWriter();
    const result = runReactFirstHook(
      { ...baseHookInput, isSubagent: true },
      writer,
      fixedNow,
    );
    expect(result).toEqual({ kind: 'skipped', skipReason: 'subagent' });
    expect(payloads).toHaveLength(0);
  });

  it('skips with `scheduled-task` reason on isScheduledTask', () => {
    const { writer, payloads } = makeRecordingWriter();
    const result = runReactFirstHook(
      { ...baseHookInput, isScheduledTask: true },
      writer,
      fixedNow,
    );
    expect(result).toEqual({ kind: 'skipped', skipReason: 'scheduled-task' });
    expect(payloads).toHaveLength(0);
  });

  it('skips with `scheduled-task-prompt-wrap` reason on wrapped prompt', () => {
    const { writer, payloads } = makeRecordingWriter();
    const result = runReactFirstHook(
      {
        ...baseHookInput,
        prompt: '[SCHEDULED TASK] body',
      },
      writer,
      fixedNow,
    );
    expect(result).toEqual({
      kind: 'skipped',
      skipReason: 'scheduled-task-prompt-wrap',
    });
    expect(payloads).toHaveLength(0);
  });

  it('skips with `no-assistant-name` reason when assistantName is empty', () => {
    const { writer, payloads } = makeRecordingWriter();
    const result = runReactFirstHook(
      { ...baseHookInput, assistantName: '' },
      writer,
      fixedNow,
    );
    expect(result).toEqual({
      kind: 'skipped',
      skipReason: 'no-assistant-name',
    });
    expect(payloads).toHaveLength(0);
  });

  it('returns `ipc-failed` on an expected NodeJS.ErrnoException', () => {
    const writer: ReactIpcWriter = () => {
      const err = new Error('write EROFS') as NodeJS.ErrnoException;
      err.code = 'EROFS';
      throw err;
    };
    const result = runReactFirstHook(baseHookInput, writer, fixedNow);
    expect(result).toEqual({
      kind: 'ipc-failed',
      emoji: '👀',
      code: 'EROFS',
      message: 'write EROFS',
    });
  });

  it('returns `ipc-failed` on ENOSPC (disk full)', () => {
    const writer: ReactIpcWriter = () => {
      const err = new Error('no space left on device') as NodeJS.ErrnoException;
      err.code = 'ENOSPC';
      throw err;
    };
    const result = runReactFirstHook(baseHookInput, writer, fixedNow);
    expect(result.kind).toBe('ipc-failed');
    if (result.kind === 'ipc-failed') {
      expect(result.code).toBe('ENOSPC');
    }
  });

  it('rethrows non-errno errors so programming bugs surface', () => {
    const writer: ReactIpcWriter = () => {
      throw new TypeError('bad payload shape');
    };
    expect(() => runReactFirstHook(baseHookInput, writer, fixedNow)).toThrow(
      TypeError,
    );
  });

  it('rethrows errno errors with codes outside the allow-list', () => {
    const writer: ReactIpcWriter = () => {
      const err = new Error('busy') as NodeJS.ErrnoException;
      err.code = 'EBUSY';
      throw err;
    };
    expect(() => runReactFirstHook(baseHookInput, writer, fixedNow)).toThrow();
  });

  it('builds payloads with the supplied chatJid / sessionName / groupFolder', () => {
    const { writer, payloads } = makeRecordingWriter();
    runReactFirstHook(
      {
        ...baseHookInput,
        chatJid: 'other@s.whatsapp.net',
        sessionName: 'maintenance',
        groupFolder: 'another-group',
      },
      writer,
      fixedNow,
    );
    expect(payloads[0].chatJid).toBe('other@s.whatsapp.net');
    expect(payloads[0].sessionName).toBe('maintenance');
    expect(payloads[0].groupFolder).toBe('another-group');
  });

  // #289 follow-up — pin the IPC reaction to the triggering message
  // so a slow spawn + a newer inbound between routing and hook fire
  // can't race the host's `reactToLatestMessage` fallback into
  // marking the wrong message.
  it('stamps the triggering messageId on the payload when supplied', () => {
    const { writer, payloads } = makeRecordingWriter();
    runReactFirstHook(
      { ...baseHookInput, messageId: 'msg_42' },
      writer,
      fixedNow,
    );
    expect(payloads[0].messageId).toBe('msg_42');
  });

  it('omits messageId on the payload when not supplied (legacy / scheduled paths)', () => {
    const { writer, payloads } = makeRecordingWriter();
    runReactFirstHook(baseHookInput, writer, fixedNow);
    expect(payloads[0]).not.toHaveProperty('messageId');
  });
});
