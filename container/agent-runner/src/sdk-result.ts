import type { SDKResultMessage } from '@anthropic-ai/claude-agent-sdk';

export interface ContainerOutput {
  status: 'success' | 'error';
  result: string | null;
  newSessionId?: string;
  error?: string;
  streamText?: string;
  // Successful SDK turn completion, not a host/channel delivery acknowledgment.
  completion?: 'sdk_turn_completed' | 'script_precheck_completed';
}

/** Called only for the SDK's actual result event, never preview/session updates. */
export function sdkResultOutput(
  message: SDKResultMessage,
  newSessionId?: string,
): ContainerOutput {
  if (message.subtype !== 'success' || message.is_error) {
    return {
      status: 'error',
      result: null,
      newSessionId,
      error: `SDK query did not complete successfully (${message.subtype})`,
    };
  }
  return {
    status: 'success',
    result: message.result || null,
    newSessionId,
    completion: 'sdk_turn_completed',
  };
}
