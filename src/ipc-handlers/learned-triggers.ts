import fs from 'fs';

import { registerIpcHandler, scriptResultPath } from '../ipc-registry.js';
import { logger } from '../logger.js';
import type { RegisteredGroup, TriggerPattern } from '../types.js';

/**
 * Learned-trigger lifecycle commands (#845 slice 4): the #451 dashboard
 * read plus promote / re-enable / delete on trigger-learner proposals.
 * All four use owner-of-the-bill authorization (main can target any
 * group; a non-main caller only its own folder), gated inside each
 * handler.
 */
export function registerLearnedTriggerIpcHandlers(): void {
  registerIpcHandler('list_learned_triggers', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Read-side surface for #451 item 3. Returns the learned-source
      // patterns with their observability fields (precision,
      // sample_count, last_matched_at, last_updated_at, pattern_version,
      // proposed_at, enabled, disabled). One group with `groupFolder`
      // set; all groups when omitted (main only). Authorization:
      // owner-of-bill — non-main can only inspect its own folder.
      // Result shape (matches list_installed_tiles convention):
      //   { stdout: JSON.stringify({ groups: [{folder, name, learned:[...]}] }) }
      const resultPath = scriptResultPath(sourceGroup, data);
      const filterFolder =
        typeof data.groupFolder === 'string' && data.groupFolder.trim()
          ? data.groupFolder.trim()
          : null;
      if (!isMain && filterFolder !== null && filterFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, requested: filterFolder },
          'Unauthorized list_learned_triggers attempt blocked',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: `list_learned_triggers: cross-folder read denied — non-main caller "${sourceGroup}" cannot inspect "${filterFolder}"`,
          }),
        );
        return;
      }
      // Non-main without filter is implicitly scoped to its own folder.
      const effectiveFolder =
        !isMain && filterFolder === null ? sourceGroup : filterFolder;
      const out: Array<{
        folder: string;
        name: string;
        learned: TriggerPattern[];
      }> = [];
      for (const [, g] of Object.entries(registeredGroups)) {
        if (effectiveFolder !== null && g.folder !== effectiveFolder) continue;
        const patterns = g.triggerPatterns?.patterns ?? [];
        const learned = patterns.filter((p) => p.source === 'learned');
        out.push({ folder: g.folder, name: g.name, learned });
      }
      fs.writeFileSync(
        resultPath,
        JSON.stringify({ stdout: JSON.stringify({ groups: out }) }),
      );
      logger.info(
        {
          sourceGroup,
          scope: effectiveFolder ?? 'all',
          group_count: out.length,
          learned_total: out.reduce((n, g) => n + g.learned.length, 0),
        },
        'list_learned_triggers served via IPC',
      );
    },
  });

  registerIpcHandler('promote_learned_trigger', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Promotion path for #451 item 1. The trigger-pattern learner
      // (#415, src/gates/trigger-learner.ts) writes proposals with
      // `source: 'learned'` and `enabled: false`. The trigger gate
      // skips `enabled: false` rows so proposals are inert until the
      // operator promotes them. This handler flips `enabled: false →
      // true` on a specific learned proposal identified by its
      // `{kind, pattern}` tuple. Demotion / re-enable and the
      // per-group dashboard are #451 items 2/3 — separate scopes.
      // Authorization mirrors set_agent_model: owner-of-bill —
      // non-main can only promote in its own folder.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      const kind = typeof data.kind === 'string' ? data.kind.trim() : '';
      // Pattern is the identity field used to match the row in
      // triggerPatterns — trimming would break lookups for patterns
      // legitimately stored with significant whitespace. Reject
      // whitespace-only / empty as a separate validation step rather
      // than mutating the value before the lookup.
      const pattern = typeof data.pattern === 'string' ? data.pattern : '';
      if (!groupFolder || !kind || !pattern || pattern.trim().length === 0) {
        logger.warn(
          { data },
          'Invalid promote_learned_trigger request - groupFolder, kind, and pattern are all required strings',
        );
        return;
      }
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'promote_learned_trigger: group not registered (use register_group first)',
        );
        return;
      }
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized promote_learned_trigger attempt blocked',
        );
        return;
      }
      const cfg = targetGroup.triggerPatterns;
      if (!cfg) {
        logger.warn(
          { groupFolder },
          'promote_learned_trigger: group has no triggerPatterns config (no learned proposals to promote)',
        );
        return;
      }
      const idx = cfg.patterns.findIndex(
        (p) =>
          p.source === 'learned' && p.kind === kind && p.pattern === pattern,
      );
      if (idx === -1) {
        logger.warn(
          { groupFolder, kind, pattern },
          'promote_learned_trigger: no matching learned pattern (check kind+pattern against registered_groups.trigger_pattern JSON)',
        );
        return;
      }
      const target = cfg.patterns[idx];
      if (target.disabled === true) {
        logger.warn(
          { groupFolder, kind, pattern },
          'promote_learned_trigger: pattern is auto-rolled-back (disabled=true) — re-enable via the demotion handler before promoting (item 2)',
        );
        return;
      }
      if (target.enabled === true) {
        logger.info(
          { groupFolder, kind, pattern, source: sourceGroup },
          'promote_learned_trigger: pattern already enabled, no-op',
        );
        return;
      }
      const updatedPatterns = cfg.patterns.map((p, i) =>
        i === idx ? { ...p, enabled: true } : p,
      );
      deps.registerGroup(targetJid, {
        ...targetGroup,
        triggerPatterns: { ...cfg, patterns: updatedPatterns },
      });
      logger.info(
        {
          groupFolder,
          kind,
          pattern,
          pattern_version: target.pattern_version ?? null,
          precision: target.precision,
          sample_count: target.sample_count,
          source: sourceGroup,
        },
        'promote_learned_trigger: flipped enabled false → true',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
    },
  });

  registerIpcHandler('reenable_learned_trigger', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Re-enable path for #451 item 2. Counterpart to the learner's
      // auto-rollback (`disabled: true`) — flips `disabled: true →
      // false` so the matcher resumes consuming the row. Operator's
      // remit: they've fixed the root cause (e.g. removed a noisy
      // synthetic-identity nickname) and want the proposal active
      // again WITHOUT bumping pattern_version. The learner is the
      // sole writer that ever sets `disabled: true`, so re-enable is
      // by definition operator-initiated. Same `{kind, pattern}`
      // identity + same authorization shape as promote.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      const kind = typeof data.kind === 'string' ? data.kind.trim() : '';
      // Pattern is the identity field used to match the row in
      // triggerPatterns — trimming would break lookups for patterns
      // legitimately stored with significant whitespace. Reject
      // whitespace-only / empty as a separate validation step rather
      // than mutating the value before the lookup.
      const pattern = typeof data.pattern === 'string' ? data.pattern : '';
      if (!groupFolder || !kind || !pattern || pattern.trim().length === 0) {
        logger.warn(
          { data },
          'Invalid reenable_learned_trigger request - groupFolder, kind, and pattern are all required strings',
        );
        return;
      }
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'reenable_learned_trigger: group not registered',
        );
        return;
      }
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized reenable_learned_trigger attempt blocked',
        );
        return;
      }
      const cfg = targetGroup.triggerPatterns;
      if (!cfg) {
        logger.warn(
          { groupFolder },
          'reenable_learned_trigger: group has no triggerPatterns config',
        );
        return;
      }
      const idx = cfg.patterns.findIndex(
        (p) =>
          p.source === 'learned' && p.kind === kind && p.pattern === pattern,
      );
      if (idx === -1) {
        logger.warn(
          { groupFolder, kind, pattern },
          'reenable_learned_trigger: no matching learned pattern',
        );
        return;
      }
      const target = cfg.patterns[idx];
      if (target.disabled !== true) {
        logger.info(
          { groupFolder, kind, pattern, source: sourceGroup },
          'reenable_learned_trigger: pattern was not disabled, no-op',
        );
        return;
      }
      const updatedPatterns = cfg.patterns.map((p, i) =>
        i === idx ? { ...p, disabled: false } : p,
      );
      deps.registerGroup(targetJid, {
        ...targetGroup,
        triggerPatterns: { ...cfg, patterns: updatedPatterns },
      });
      logger.info(
        {
          groupFolder,
          kind,
          pattern,
          pattern_version: target.pattern_version ?? null,
          source: sourceGroup,
        },
        'reenable_learned_trigger: flipped disabled true → false',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
    },
  });

  registerIpcHandler('delete_learned_trigger', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Permanent-delete path for #451 item 2 (second half). Removes
      // a learned proposal from the array entirely — for proposals
      // that should never come back regardless of root-cause fixes
      // (e.g. a pattern that overlaps with an owner-set entry, or a
      // demoted row whose precision is irrecoverable). Distinct from
      // re-enable: deletion is final and the learner can re-propose
      // the same body later as a fresh row with `pattern_version: 1`.
      // Same authorization shape as promote/reenable.
      const groupFolder =
        typeof data.groupFolder === 'string' ? data.groupFolder.trim() : '';
      const kind = typeof data.kind === 'string' ? data.kind.trim() : '';
      // Pattern is the identity field used to match the row in
      // triggerPatterns — trimming would break lookups for patterns
      // legitimately stored with significant whitespace. Reject
      // whitespace-only / empty as a separate validation step rather
      // than mutating the value before the lookup.
      const pattern = typeof data.pattern === 'string' ? data.pattern : '';
      if (!groupFolder || !kind || !pattern || pattern.trim().length === 0) {
        logger.warn(
          { data },
          'Invalid delete_learned_trigger request - groupFolder, kind, and pattern are all required strings',
        );
        return;
      }
      let targetJid: string | undefined;
      let targetGroup: RegisteredGroup | undefined;
      for (const [jid, g] of Object.entries(registeredGroups)) {
        if (g.folder === groupFolder) {
          targetJid = jid;
          targetGroup = g;
          break;
        }
      }
      if (!targetJid || !targetGroup) {
        logger.warn(
          { groupFolder },
          'delete_learned_trigger: group not registered',
        );
        return;
      }
      if (!isMain && groupFolder !== sourceGroup) {
        logger.warn(
          { sourceGroup, groupFolder },
          'Unauthorized delete_learned_trigger attempt blocked',
        );
        return;
      }
      const cfg = targetGroup.triggerPatterns;
      if (!cfg) {
        logger.warn(
          { groupFolder },
          'delete_learned_trigger: group has no triggerPatterns config',
        );
        return;
      }
      const before = cfg.patterns.length;
      const updatedPatterns = cfg.patterns.filter(
        (p) =>
          !(p.source === 'learned' && p.kind === kind && p.pattern === pattern),
      );
      if (updatedPatterns.length === before) {
        logger.warn(
          { groupFolder, kind, pattern },
          'delete_learned_trigger: no matching learned pattern',
        );
        return;
      }
      deps.registerGroup(targetJid, {
        ...targetGroup,
        triggerPatterns: { ...cfg, patterns: updatedPatterns },
      });
      logger.info(
        {
          groupFolder,
          kind,
          pattern,
          remaining_patterns: updatedPatterns.length,
          source: sourceGroup,
        },
        'delete_learned_trigger: removed learned proposal',
      );
      const availableGroups = deps.getAvailableGroups();
      deps.writeGroupsSnapshot(
        sourceGroup,
        isMain,
        availableGroups,
        new Set(Object.keys(registeredGroups)),
      );
    },
  });
}
