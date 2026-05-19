import { describe, expect, it } from 'vitest';
import {
  buildSetAgentModelPayload,
  buildSetMaintenanceAgentModelPayload,
  buildSetTaskAgentModelPayload,
  describeAgentModelChange,
  normalizeAgentModelInput,
} from './agent-model-payload.js';

const FIXED_NOW = new Date('2026-05-19T20:30:00.000Z');

describe('normalizeAgentModelInput', () => {
  it('passes null through as null', () => {
    expect(normalizeAgentModelInput(null)).toBeNull();
  });

  it('collapses empty string to null', () => {
    expect(normalizeAgentModelInput('')).toBeNull();
  });

  it('collapses whitespace-only string to null', () => {
    expect(normalizeAgentModelInput('   ')).toBeNull();
    expect(normalizeAgentModelInput('\t')).toBeNull();
    expect(normalizeAgentModelInput('\n  \t')).toBeNull();
  });

  it('trims surrounding whitespace on non-empty values', () => {
    expect(normalizeAgentModelInput('  claude-haiku-4-5-20251001  ')).toBe(
      'claude-haiku-4-5-20251001',
    );
    expect(normalizeAgentModelInput('claude-sonnet-4-6\n')).toBe(
      'claude-sonnet-4-6',
    );
  });

  it('leaves non-padded values byte-identical', () => {
    expect(normalizeAgentModelInput('claude-sonnet-4-6')).toBe(
      'claude-sonnet-4-6',
    );
  });
});

describe('buildSetAgentModelPayload', () => {
  it('builds the IPC payload with normalized model', () => {
    expect(
      buildSetAgentModelPayload(
        { groupFolder: 'telegram_swarm', agentModel: 'claude-sonnet-4-6' },
        FIXED_NOW,
      ),
    ).toEqual({
      type: 'set_agent_model',
      groupFolder: 'telegram_swarm',
      agentModel: 'claude-sonnet-4-6',
      timestamp: '2026-05-19T20:30:00.000Z',
    });
  });

  it('collapses whitespace-padded model to its trimmed form', () => {
    expect(
      buildSetAgentModelPayload(
        { groupFolder: 'telegram_swarm', agentModel: '  haiku  ' },
        FIXED_NOW,
      ).agentModel,
    ).toBe('haiku');
  });

  it('collapses whitespace-only model to null (clear)', () => {
    expect(
      buildSetAgentModelPayload(
        { groupFolder: 'telegram_swarm', agentModel: '   ' },
        FIXED_NOW,
      ).agentModel,
    ).toBeNull();
  });

  it('passes explicit null through as a clear', () => {
    expect(
      buildSetAgentModelPayload(
        { groupFolder: 'telegram_swarm', agentModel: null },
        FIXED_NOW,
      ).agentModel,
    ).toBeNull();
  });
});

describe('buildSetMaintenanceAgentModelPayload', () => {
  it('builds the IPC payload with the maintenance-specific field name', () => {
    expect(
      buildSetMaintenanceAgentModelPayload(
        {
          groupFolder: 'telegram_swarm',
          maintenanceAgentModel: 'claude-sonnet-4-6',
        },
        FIXED_NOW,
      ),
    ).toEqual({
      type: 'set_maintenance_agent_model',
      groupFolder: 'telegram_swarm',
      maintenanceAgentModel: 'claude-sonnet-4-6',
      timestamp: '2026-05-19T20:30:00.000Z',
    });
  });

  it('normalizes the maintenance model field', () => {
    expect(
      buildSetMaintenanceAgentModelPayload(
        { groupFolder: 'telegram_swarm', maintenanceAgentModel: '  ' },
        FIXED_NOW,
      ).maintenanceAgentModel,
    ).toBeNull();
    expect(
      buildSetMaintenanceAgentModelPayload(
        {
          groupFolder: 'telegram_swarm',
          maintenanceAgentModel: '  haiku  ',
        },
        FIXED_NOW,
      ).maintenanceAgentModel,
    ).toBe('haiku');
  });
});

describe('buildSetTaskAgentModelPayload', () => {
  it('maps snake_case task_id to camelCase taskId in the IPC payload', () => {
    expect(
      buildSetTaskAgentModelPayload(
        { task_id: 'task-abc', agentModel: 'claude-haiku-4-5-20251001' },
        FIXED_NOW,
      ),
    ).toEqual({
      type: 'set_task_agent_model',
      taskId: 'task-abc',
      agentModel: 'claude-haiku-4-5-20251001',
      timestamp: '2026-05-19T20:30:00.000Z',
    });
  });

  it('normalizes the model field for tasks', () => {
    expect(
      buildSetTaskAgentModelPayload(
        { task_id: 'task-abc', agentModel: '   ' },
        FIXED_NOW,
      ).agentModel,
    ).toBeNull();
    expect(
      buildSetTaskAgentModelPayload(
        { task_id: 'task-abc', agentModel: '\tclaude-sonnet-4-6  ' },
        FIXED_NOW,
      ).agentModel,
    ).toBe('claude-sonnet-4-6');
  });

  it('passes null through as a clear', () => {
    expect(
      buildSetTaskAgentModelPayload(
        { task_id: 'task-abc', agentModel: null },
        FIXED_NOW,
      ).agentModel,
    ).toBeNull();
  });
});

describe('describeAgentModelChange', () => {
  it('quotes a non-null value', () => {
    expect(describeAgentModelChange('haiku', 'cleared')).toBe('"haiku"');
  });

  it('returns the fallback description for null', () => {
    expect(describeAgentModelChange(null, 'cleared (use global default)')).toBe(
      'cleared (use global default)',
    );
  });
});
