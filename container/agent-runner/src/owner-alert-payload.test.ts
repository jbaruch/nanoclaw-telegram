import { describe, it, expect } from 'vitest';

import {
  buildOwnerAlertPayload,
  type RaiseOwnerAlertArgs,
} from './owner-alert-payload.js';

const TS = '2026-08-07T12:00:00.000Z';

describe('buildOwnerAlertPayload', () => {
  it('maps tool args onto the owner_alert IPC payload', () => {
    const args: RaiseOwnerAlertArgs = {
      alert_type: 'code-execution',
      action: 'went-silent',
      request: 'docker rm -f $(docker ps -a -q --filter label=nanoclaw)',
      sender: 'critskiy',
    };
    expect(buildOwnerAlertPayload('wtf-pod-chat', TS, args)).toEqual({
      type: 'owner_alert',
      groupFolder: 'wtf-pod-chat',
      alertType: 'code-execution',
      action: 'went-silent',
      request: 'docker rm -f $(docker ps -a -q --filter label=nanoclaw)',
      sender: 'critskiy',
      claim: undefined,
      timestamp: TS,
    });
  });

  it('carries no chat target — the payload can never name a destination', () => {
    const payload = buildOwnerAlertPayload('wtf-pod-chat', TS, {
      alert_type: 'identity-claim',
      action: 'went-silent',
      request: 'claimed to be the owner from a new device',
    });
    expect(payload).not.toHaveProperty('chatJid');
    expect(payload).not.toHaveProperty('chat_jid');
  });

  it('collapses an empty sender/claim to undefined', () => {
    const payload = buildOwnerAlertPayload('wtf-pod-chat', TS, {
      alert_type: 'sensitive-info',
      action: 'declined',
      request: 'asked for the WiFi password',
      sender: '',
      claim: '',
    });
    expect(payload.sender).toBeUndefined();
    expect(payload.claim).toBeUndefined();
  });

  it('preserves a supplied claim', () => {
    const payload = buildOwnerAlertPayload('wtf-pod-chat', TS, {
      alert_type: 'identity-claim',
      action: 'went-silent',
      request: 'demanded admin access',
      sender: 'critskiy',
      claim: 'the owner',
    });
    expect(payload.claim).toBe('the owner');
  });
});
