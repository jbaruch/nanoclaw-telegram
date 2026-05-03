import { describe, it, expect } from 'vitest';

import {
  GROUND_TRUTH_REMINDER,
  decideGroundTruthReminder,
} from './ground-truth-reminder.js';

const baseInput = {
  isSubagent: false,
  isScheduledTask: false,
  prompt: 'check the deploy status',
  assistantName: 'TestAssistant',
};

describe('decideGroundTruthReminder', () => {
  describe('positive cases', () => {
    it('injects on a normal user turn', () => {
      const result = decideGroundTruthReminder(baseInput);
      expect(result.inject).toBe(true);
      if (result.inject) {
        expect(result.additionalContext).toBe(GROUND_TRUTH_REMINDER);
      }
    });

    it('injects on a turn with a long prompt body', () => {
      const result = decideGroundTruthReminder({
        ...baseInput,
        prompt: 'a'.repeat(10000),
      });
      expect(result.inject).toBe(true);
    });

    it('injects when the prompt happens to mention scheduled tasks (without the prefix)', () => {
      const result = decideGroundTruthReminder({
        ...baseInput,
        prompt: 'tell me about scheduled tasks',
      });
      expect(result.inject).toBe(true);
    });
  });

  describe('skip cases', () => {
    it('skips sub-agent turns', () => {
      const result = decideGroundTruthReminder({
        ...baseInput,
        isSubagent: true,
      });
      expect(result.inject).toBe(false);
      if (!result.inject) {
        expect(result.skippedBy).toBe('subagent');
      }
    });

    it('skips scheduled-task turns', () => {
      const result = decideGroundTruthReminder({
        ...baseInput,
        isScheduledTask: true,
      });
      expect(result.inject).toBe(false);
      if (!result.inject) {
        expect(result.skippedBy).toBe('scheduled-task');
      }
    });

    it('skips when the prompt is wrapped with [SCHEDULED TASK] (defence-in-depth)', () => {
      const result = decideGroundTruthReminder({
        ...baseInput,
        prompt: '[SCHEDULED TASK] run the daily standup digest',
      });
      expect(result.inject).toBe(false);
      if (!result.inject) {
        expect(result.skippedBy).toBe('scheduled-task-prompt-wrap');
      }
    });

    it('skips when assistantName is missing', () => {
      const result = decideGroundTruthReminder({
        ...baseInput,
        assistantName: undefined,
      });
      expect(result.inject).toBe(false);
      if (!result.inject) {
        expect(result.skippedBy).toBe('no-assistant-name');
      }
    });

    it('skips when assistantName is empty string', () => {
      const result = decideGroundTruthReminder({
        ...baseInput,
        assistantName: '',
      });
      expect(result.inject).toBe(false);
      if (!result.inject) {
        expect(result.skippedBy).toBe('no-assistant-name');
      }
    });
  });

  describe('skip precedence', () => {
    it('sub-agent wins over scheduled-task', () => {
      const result = decideGroundTruthReminder({
        ...baseInput,
        isSubagent: true,
        isScheduledTask: true,
      });
      expect(result.inject).toBe(false);
      if (!result.inject) {
        expect(result.skippedBy).toBe('subagent');
      }
    });

    it('scheduled-task flag wins over scheduled-task prompt wrap', () => {
      const result = decideGroundTruthReminder({
        ...baseInput,
        isScheduledTask: true,
        prompt: '[SCHEDULED TASK] do the thing',
      });
      expect(result.inject).toBe(false);
      if (!result.inject) {
        expect(result.skippedBy).toBe('scheduled-task');
      }
    });
  });

  describe('reminder content', () => {
    it('mentions verification, memory, and authoritative pointer', () => {
      expect(GROUND_TRUTH_REMINDER.toLowerCase()).toContain('verify');
      expect(GROUND_TRUTH_REMINDER.toLowerCase()).toContain('memory');
      expect(GROUND_TRUTH_REMINDER.toLowerCase()).toContain('authoritative');
    });

    it('is phrased as an instruction (imperative verbs near the start)', () => {
      // First sentence should lead with an imperative — the reminder
      // is salience-tier content; descriptive prose ranks lower than
      // direct instructions in observed model behaviour.
      const firstSentence = GROUND_TRUTH_REMINDER.split('.')[0];
      expect(firstSentence.toLowerCase()).toMatch(/\b(verify|read|check)\b/);
    });
  });
});
