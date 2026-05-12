import type { StateMigration } from '../db.js';

import { STATE_001_ORDERS } from './state-001-orders.js';
import { STATE_002_EMAIL_FEEDBACK } from './state-002-email-feedback.js';
import { STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION } from './state-003-email-feedback-schema-version.js';
import { STATE_004_SCHEDULED_REMINDERS } from './state-004-scheduled-reminders.js';
import { STATE_005_NANOCLAW_STATE_SPLIT } from './state-005-nanoclaw-state-split.js';
import { STATE_006_TRUSTED_SESSION_STATE } from './state-006-trusted-session-state.js';
import { STATE_007_MORNING_BRIEF_PENDING } from './state-007-morning-brief-pending.js';
import { STATE_008_CALENDAR_STATE } from './state-008-calendar-state.js';
import { STATE_009_PHASE_COMPLETIONS } from './state-009-phase-completions.js';
import { STATE_010_TZ_STATE } from './state-010-tz-state.js';
import { STATE_011_SESSION_LENGTH_CAP } from './state-011-session-length-cap.js';
import { STATE_012_TZ_STATE_SEGMENTS } from './state-012-tz-state-segments.js';
import { STATE_013_TZ_STATE_SEGMENTS_DATETIME } from './state-013-tz-state-segments-datetime.js';

/**
 * Registered state-table migrations, applied in order at orchestrator
 * startup. Versions must be contiguous integers starting at 1 — see
 * README.md for the full convention.
 *
 * Invariants enforced by `validateMigrationRegistry` in db.ts:
 *   - Each entry's `version` equals its array index + 1
 *   - `name` is a non-empty human-readable label
 *   - `sql` is a non-empty SQL string (DDL/DML)
 */
export const STATE_MIGRATIONS: readonly StateMigration[] = [
  STATE_001_ORDERS,
  STATE_002_EMAIL_FEEDBACK,
  STATE_003_EMAIL_FEEDBACK_SCHEMA_VERSION,
  STATE_004_SCHEDULED_REMINDERS,
  STATE_005_NANOCLAW_STATE_SPLIT,
  STATE_006_TRUSTED_SESSION_STATE,
  STATE_007_MORNING_BRIEF_PENDING,
  STATE_008_CALENDAR_STATE,
  STATE_009_PHASE_COMPLETIONS,
  STATE_010_TZ_STATE,
  STATE_011_SESSION_LENGTH_CAP,
  STATE_012_TZ_STATE_SEGMENTS,
  STATE_013_TZ_STATE_SEGMENTS_DATETIME,
];
