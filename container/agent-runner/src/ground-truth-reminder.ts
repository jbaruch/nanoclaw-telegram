/**
 * ground-truth-reminder — pure decision logic for the
 * `ground-truth-reminder` UserPromptSubmit hook (#227, tracks #214).
 *
 * The agent has SOUL re-injected via the system prompt every turn,
 * but the verification rules (`ground-truth`, `read-full-content`,
 * `verification-protocol`) load once via the tile and fade in salience
 * over a long session. SOUL's idioms optimise for OUTPUT SHAPE
 * (punchline before explanation, lead with the fix, snarky-confident);
 * the verification rules optimise for OUTPUT TRUTH. Under uncertainty
 * the asymmetry biases the model toward shape over truth — the answer
 * that LOOKS right wins over the answer that IS right.
 *
 * This hook re-injects a tight ground-truth reminder on every turn,
 * raising verification discipline to the same per-turn salience tier
 * SOUL enjoys. The reminder is short by design: a long block of text
 * would crowd out the prompt-relevant content and trigger token-budget
 * concerns. One sentence is enough to cue the discipline if the agent
 * is going to apply it at all.
 *
 * Skip cases mirror `react-first.ts`:
 *  - Sub-agent turns have their own discipline (and their own context).
 *  - Scheduled-task turns are non-interactive — the reminder lands as
 *    noise the user never sees.
 *  - The `[SCHEDULED TASK]` prompt wrap is the defence-in-depth case
 *    for scheduled-task containers misreporting `isScheduledTask=false`.
 *  - Containers with no `assistantName` are bare runner / probe
 *    contexts (no persona bound to the session); skipping there
 *    keeps non-interactive runs uncluttered with per-turn salience
 *    content they don't act on.
 *
 * Kept SDK-free so the root vitest can exercise it without spinning up
 * `@anthropic-ai/claude-agent-sdk`.
 */

// Re-exported from `react-first.ts` so the wrap-skip signal is one
// canonical string. Both hooks short-circuit on the same value, with
// no drift if the wrap text is ever changed.
import { SCHEDULED_TASK_PROMPT_PREFIX } from './react-first.js';

export interface GroundTruthReminderInput {
  /** True iff this is a sub-agent (Task tool) sub-turn. */
  isSubagent: boolean;
  /** True iff the orchestrator dispatched this as a scheduled task. */
  isScheduledTask: boolean;
  /** The user-submitted prompt body. */
  prompt: string;
  /** Persona name (e.g. 'AyeAye'); empty in bare runner contexts. */
  assistantName?: string;
}

export type GroundTruthSkipReason =
  | 'subagent'
  | 'scheduled-task'
  | 'scheduled-task-prompt-wrap'
  | 'no-assistant-name';

export type GroundTruthReminderResult =
  | { inject: true; additionalContext: string }
  | { inject: false; skippedBy: GroundTruthSkipReason };

/**
 * The reminder text. Pinned to a single constant so wording changes
 * land in one place — the reminder is salience-tier content, and
 * accidental drift across call sites would dilute it.
 *
 * Phrased as an instruction, not a description: the model honours
 * imperatives more reliably than soft hedges.
 */
export const GROUND_TRUTH_REMINDER =
  'Before answering: verify any factual claim. Memory is not a source. ' +
  'If you have an authoritative pointer (memory file, env var, hardcoded ' +
  'constant), read it first. If you can verify it, you must verify it.';

/**
 * Inspect a UserPromptSubmit context and decide whether to inject the
 * ground-truth reminder into `additionalContext`.
 *
 * The decision is pure: the caller in `index.ts` plumbs SDK shape +
 * container input into `GroundTruthReminderInput`, and threads the
 * result back into the hook return value.
 */
export function decideGroundTruthReminder(
  input: GroundTruthReminderInput,
): GroundTruthReminderResult {
  if (input.isSubagent) {
    return { inject: false, skippedBy: 'subagent' };
  }
  if (input.isScheduledTask) {
    return { inject: false, skippedBy: 'scheduled-task' };
  }
  if (input.prompt.startsWith(SCHEDULED_TASK_PROMPT_PREFIX)) {
    return { inject: false, skippedBy: 'scheduled-task-prompt-wrap' };
  }
  if (!input.assistantName || input.assistantName.length === 0) {
    return { inject: false, skippedBy: 'no-assistant-name' };
  }
  return { inject: true, additionalContext: GROUND_TRUTH_REMINDER };
}
