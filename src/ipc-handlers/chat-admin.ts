import fs from 'fs';

import { ASSISTANT_NAME } from '../config.js';
import { sendPoolMessage } from '../channels/telegram.js';
import {
  resolveAgentModel,
  resolvePerGroupAgentModel,
} from '../container-runner.js';
import {
  getLastFromMeMessages,
  shouldStoreBotMessage,
  storeMessage,
} from '../db-messages.js';
import { findGateDecisions, readHostLog } from '../host-log-parser.js';
import { hostLogsOrchestratorFile } from '../host-logs.js';
import { registerIpcHandler, scriptResultPath } from '../ipc-registry.js';
import { logger } from '../logger.js';
import { stripInternalTags } from '../router.js';

/**
 * Chat-administration commands (#845 slice 5): the per-chat nuke plus
 * the admin tile's cross-chat surface (status snapshot, gate-decision
 * inspection, cross-chat nuke, cross-chat broadcast). The four admin
 * commands write their error envelope UNCONDITIONALLY — including to
 * the orphan result path for a payload with a bad requestId — so they
 * keep their own in-handler isMain gate instead of the dispatcher's
 * `requiresMain` (which only writes an envelope for valid requestIds).
 */
export function registerChatAdminIpcHandlers(): void {
  registerIpcHandler('nuke_session', {
    handler: ({ data, sourceGroup, deps }) => {
      if (data.groupFolder) {
        // Optional `session` arg narrows the nuke to one slot. Accepted
        // values: 'default', 'maintenance', 'all'. Anything else (or
        // missing) falls back to 'all' — the safe default that preserves
        // pre-parallel behaviour. The value comes from the container's
        // IPC payload so we cast from `unknown` and allowlist.
        const sessionArg = (data as unknown as Record<string, unknown>).session;
        const validSession: 'default' | 'maintenance' | 'all' =
          sessionArg === 'default' || sessionArg === 'maintenance'
            ? sessionArg
            : 'all';
        // Optional `skipReentry` (#127): when true, the dispatcher also
        // deletes `.checkpoints/default.md` + `previous.md` after the
        // standard wipe so the next spawn has no reentry Facts to load.
        // Strict boolean check — anything non-true (missing, null, the
        // string "true", etc.) falls back to the safe default of
        // preserving the checkpoint, so a malformed payload can't
        // accidentally erase reentry state.
        const skipReentryArg = (data as unknown as Record<string, unknown>)
          .skipReentry;
        const skipReentry = skipReentryArg === true;
        // `sourceGroup` is authoritative (derived from the IPC dir the
        // request arrived in); `data.groupFolder` is only used as a
        // "yes-really-nuke" opt-in flag above and its value isn't honoured
        // downstream. Log sourceGroup to avoid misleading audit trails if
        // they ever differ.
        logger.info(
          { sourceGroup, session: validSession, skipReentry },
          'Session nuke requested via IPC',
        );
        deps.nukeSession(sourceGroup, validSession, { skipReentry });
      }
    },
  });

  registerIpcHandler('chat_status', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Admin tile only. Returns a structured snapshot per chat: the
      // host-side state the admin needs to diagnose silent containers
      // (running / idle / cooling-down / crashed / not-spawned), tile
      // classification, trigger config, and the latest is_from_me=1
      // message recorded for the chat.
      const resultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized chat_status attempt blocked',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: 'chat_status is admin-tile only',
          }),
        );
        return;
      }

      // Resolve which chats to report on. Four cases:
      //   - both chat_id AND chat_name → reject. Two identifiers that
      //     might disagree is unsafe targeting; force the caller to
      //     pick one. Defense in depth — the MCP tool layer also
      //     blocks this, but a payload arriving directly via the IPC
      //     dir would otherwise let chat_id silently win.
      //   - chat_id provided → report only that one (must be registered).
      //   - chat_name provided → resolve via name match in
      //     registeredGroups (multiple matches → ambiguous error so the
      //     caller can pick the right JID rather than us guessing).
      //   - neither provided → all registered chats.
      const hasChatId =
        typeof data.chat_id === 'string' && data.chat_id.trim().length > 0;
      const hasChatName =
        typeof data.chat_name === 'string' && data.chat_name.trim().length > 0;
      if (hasChatId && hasChatName) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'chat_status accepts chat_id OR chat_name, not both — they may disagree',
          }),
        );
        return;
      }
      const targets: string[] = [];
      if (hasChatId) {
        const trimmed = (data.chat_id as string).trim();
        if (!registeredGroups[trimmed]) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_id ${trimmed} not registered`,
            }),
          );
          return;
        }
        targets.push(trimmed);
      } else if (hasChatName) {
        const wanted = (data.chat_name as string).trim();
        const matches = Object.entries(registeredGroups).filter(
          ([, g]) => g.name === wanted,
        );
        if (matches.length === 0) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" did not match any registered chat`,
            }),
          );
          return;
        }
        if (matches.length > 1) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" is ambiguous — matches ${matches.length} chats`,
              candidates: matches.map(([jid]) => jid),
            }),
          );
          return;
        }
        targets.push(matches[0][0]);
      } else {
        targets.push(...Object.keys(registeredGroups));
      }

      // Batch the "latest is_from_me=1 message per chat" lookup into a
      // single grouped query (idx_messages_fromme_chat composite
      // index). Per-target getLastFromMeMessage calls were N
      // statement compilations + N scan-and-sort passes; this is one
      // query for any N.
      const lastMessages = getLastFromMeMessages(targets);

      // Resolve the orchestrator-wide default once. Each row then runs
      // the per-group resolver against this baseline so the value
      // reported here matches what `runContainerAgent` actually sets
      // as `AGENT_MODEL` on spawn — including the typo-fallback
      // behavior of `resolvePerGroupAgentModel`.
      const globalDefaultAgentModel = resolveAgentModel(
        process.env.AGENT_MODEL,
      );

      const rows = targets.map((jid) => {
        const group = registeredGroups[jid];
        const tile: 'admin' | 'trusted' | 'untrusted' = group.isMain
          ? 'admin'
          : group.containerConfig?.trusted
            ? 'trusted'
            : 'untrusted';
        // requiresTrigger defaults differ per tile: main groups bypass
        // the trigger entirely (privileged inbox), while non-main groups
        // require the trigger unless explicitly opted out. Mirror the
        // canSenderInteract logic so the reported value matches what
        // the orchestrator actually enforces.
        const triggered = group.isMain
          ? false
          : group.requiresTrigger !== false;
        const last = lastMessages.get(jid) ?? null;
        return {
          chat_id: jid,
          chat_name: group.name,
          trigger: triggered ? 'triggered' : 'untriggered',
          tile,
          last_ayeaye_message: last
            ? {
                timestamp: last.timestamp,
                // Truncate to keep the response small even if the
                // agent sent a multi-kilobyte reply. 200 chars matches
                // what fits comfortably in the admin's chat preview.
                content_snippet:
                  last.content.length > 200
                    ? last.content.slice(0, 200) + '…'
                    : last.content,
              }
            : null,
          containers: deps.getContainerStatus
            ? {
                default: deps.getContainerStatus(jid, 'default'),
                maintenance: deps.getContainerStatus(jid, 'maintenance'),
              }
            : { default: 'not-spawned', maintenance: 'not-spawned' },
          // Effective AGENT_MODEL for this group's next spawn — the
          // per-group `containerConfig.agentModel` override resolved
          // against the orchestrator-wide default. Surfaced for cost
          // attribution / audit so operators don't have to grep spawn
          // logs to learn which group runs which model (#395 follow-up).
          // Mirrors the resolver at the spawn site exactly: typo-bad
          // overrides resolve to the global default here too.
          effective_agent_model: resolvePerGroupAgentModel(
            group.containerConfig?.agentModel,
            globalDefaultAgentModel,
          ),
        };
      });

      logger.info(
        { sourceGroup, count: rows.length },
        'chat_status served via IPC',
      );
      fs.writeFileSync(
        resultPath,
        JSON.stringify({ stdout: JSON.stringify({ chats: rows }) }),
      );
    },
  });

  registerIpcHandler('inspect_gate_decisions', {
    handler: ({ data, sourceGroup, isMain }) => {
      // Admin tile only. Returns the most-recent gate-decision records
      // for a chat — one row per `evaluateGateChain` per-message call,
      // captured from the canonical INFO line `'gate decision'` in
      // `data/host-logs/orchestrator.log` (#443). Logs are the
      // non-purgeable substrate the design landed on after rejecting a
      // SQLite table (every-message persistence on durable media isn't
      // worth the retention cost when logs already rotate via
      // `scripts/logrotate.sh`); the file is bind-mounted RO into the
      // admin agent container so the response payload is the same data
      // the agent could grep for itself, just structured.
      const resultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized inspect_gate_decisions attempt blocked',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: 'inspect_gate_decisions is admin-tile only',
          }),
        );
        return;
      }
      // chat_id is required — the canonical use case is "tell me about
      // chat X"; an unbounded scan over every chat is not what this
      // tool is for. Listing across chats would also require a much
      // larger limit and risk exposing cross-chat traffic in a single
      // response.
      const chatId =
        typeof data.chat_id === 'string' ? data.chat_id.trim() : '';
      if (chatId.length === 0) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: 'inspect_gate_decisions requires chat_id',
          }),
        );
        return;
      }
      // `limit` defaults to 10 — small enough that the response fits
      // comfortably in an MCP-tool text reply, large enough to cover
      // a recent burst when the user asks "what did the gate think
      // about the last few messages". Cap at 100 so a typo can't
      // request a multi-megabyte payload.
      let limit = 10;
      if (typeof data.limit === 'number' && Number.isInteger(data.limit)) {
        if (data.limit < 1) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: 'limit must be a positive integer (1–100)',
            }),
          );
          return;
        }
        limit = Math.min(data.limit, 100);
      }
      const messageId =
        typeof data.message_id === 'string' && data.message_id.length > 0
          ? data.message_id
          : undefined;
      const records = readHostLog(hostLogsOrchestratorFile());
      const hits = findGateDecisions(records, {
        chatJid: chatId,
        messageId,
        limit,
      });
      logger.info(
        {
          sourceGroup,
          chatId,
          messageId,
          limit,
          hitCount: hits.length,
        },
        'inspect_gate_decisions served via IPC',
      );
      fs.writeFileSync(
        resultPath,
        JSON.stringify({ stdout: JSON.stringify({ decisions: hits }) }),
      );
    },
  });

  registerIpcHandler('nuke_chat', {
    handler: ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Admin tile only. Cross-chat nuke — looks up the target by
      // chat_id or chat_name and forwards to the same wipeSessionJsonl
      // path the per-chat nuke_session uses. Hard-fails when neither
      // identifier is provided so admin can never accidentally nuke
      // its own chat by omission (the nuke_session tool already does
      // "this chat" — nuke_chat is only useful when targeting another).
      const resultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn({ sourceGroup }, 'Unauthorized nuke_chat attempt blocked');
        fs.writeFileSync(
          resultPath,
          JSON.stringify({ error: 'nuke_chat is admin-tile only' }),
        );
        return;
      }

      const hasId =
        typeof data.chat_id === 'string' && data.chat_id.trim().length > 0;
      const hasName =
        typeof data.chat_name === 'string' && data.chat_name.trim().length > 0;
      if (!hasId && !hasName) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'nuke_chat requires chat_id or chat_name — admin always operates cross-chat, never on the implicit current chat',
          }),
        );
        return;
      }
      // Two identifiers are an unsafe-targeting smell — if they
      // disagree, silently picking one is worse than refusing. Reject
      // here too (the MCP tool layer also blocks the same case).
      if (hasId && hasName) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'nuke_chat accepts chat_id OR chat_name, not both — they may disagree',
          }),
        );
        return;
      }

      let targetJid = '';
      if (hasId) {
        const trimmed = (data.chat_id as string).trim();
        if (!registeredGroups[trimmed]) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: `chat_id ${trimmed} not registered` }),
          );
          return;
        }
        targetJid = trimmed;
      } else {
        const wanted = (data.chat_name as string).trim();
        const matches = Object.entries(registeredGroups).filter(
          ([, g]) => g.name === wanted,
        );
        if (matches.length === 0) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" did not match any registered chat`,
            }),
          );
          return;
        }
        if (matches.length > 1) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" is ambiguous — matches ${matches.length} chats`,
              candidates: matches.map(([jid]) => jid),
            }),
          );
          return;
        }
        targetJid = matches[0][0];
      }

      const targetGroup = registeredGroups[targetJid];
      const sessionArg = data.session;
      const validSession: 'default' | 'maintenance' | 'all' =
        sessionArg === 'default' ||
        sessionArg === 'maintenance' ||
        sessionArg === 'all'
          ? sessionArg
          : 'all';

      // Snapshot pre-nuke status to determine which slots actually had
      // a live container to kill. nukeSession ALWAYS wipes JSONL on
      // disk regardless of whether anything was running; the
      // user-visible status enum (per the issue spec) reports the
      // *live-container* outcome so admin can tell whether the call
      // actually freed any resources.
      const slotsRequested: Array<'default' | 'maintenance'> =
        validSession === 'all' ? ['default', 'maintenance'] : [validSession];
      const killedSessions: Array<'default' | 'maintenance'> = [];
      const getStatus = deps.getContainerStatus;
      for (const slot of slotsRequested) {
        const wasActive =
          getStatus &&
          (getStatus(targetJid, slot) === 'running' ||
            getStatus(targetJid, slot) === 'idle');
        if (wasActive) killedSessions.push(slot);
      }

      try {
        deps.nukeSession(targetGroup.folder, validSession);
        // Per the issue's status enum: 'success' when at least one
        // live container was killed; 'noop' when nothing was running
        // (even though the on-disk wipe still happened — see the
        // pre-snapshot comment above). 'partial' is reserved for a
        // future per-slot-failure signal from nukeSession; today
        // nukeSession is fire-and-forget per slot, so we can't
        // distinguish partial failure from full success without a
        // contract change. 'error' is reported only when nukeSession
        // throws — the catch branch below.
        const status: 'success' | 'noop' =
          killedSessions.length > 0 ? 'success' : 'noop';
        logger.info(
          {
            sourceGroup,
            targetJid,
            session: validSession,
            killedSessions,
            status,
          },
          'nuke_chat completed via IPC',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            stdout: JSON.stringify({
              chat_id: targetJid,
              chat_name: targetGroup.name,
              killed_sessions: killedSessions,
              status,
            }),
          }),
        );
        // outer-boundary-process-contract (coding-policy: error-handling):
        // IPC operation handler — the agent-runner's runHostOperation reads
        // `result.error` as the tool-failure signal.
        //   - Caller's silent-failure shape: a missing `error` field reads as
        //     success, so a swallowed failure surfaces as a phantom success.
        //   - What the catch emits: the error into `result.error` (below).
        //   - Why propagation breaks the contract: an uncaught error would
        //     skip the result envelope the caller polls for.
        // eslint-disable-next-line no-catch-all/no-catch-all -- outer-boundary-process-contract
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error({ sourceGroup, targetJid, err }, 'nuke_chat failed');
        // Top-level `error` field — runHostOperation in the
        // agent-runner only treats `result.error` as a tool failure
        // and surfaces `isError: true` to the MCP caller. Burying the
        // failure inside `stdout` would make the call look like a
        // success to Claude, which would then move on as if the wipe
        // ran. Include the structured payload alongside so the admin
        // can still see what was attempted.
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: `nuke_chat failed for ${targetJid}: ${msg}`,
            chat_id: targetJid,
            chat_name: targetGroup.name,
            killed_sessions: [],
            status: 'error',
          }),
        );
      }
    },
  });

  registerIpcHandler('send_message_to_chat', {
    handler: async ({ data, sourceGroup, isMain, deps }) => {
      const registeredGroups = deps.registeredGroups();
      // Admin tile only. Cross-chat broadcast — resolves chat_id or
      // chat_name against registeredGroups, then dispatches via the
      // same channel router used by the existing 'message' IPC path
      // (pool send for sender-tagged Telegram broadcasts, direct send
      // otherwise). Replaces the schedule_task + once: now+5s kludge
      // the agent used to reach when asked "post X to #other-chat".
      //
      // Bypassing the MESSAGES_DIR fire-and-forget path is deliberate:
      // the tool's contract surfaces failure (unknown JID, ambiguous
      // name, blocked-by-user, rate-limit) synchronously to the
      // calling agent so it can retry or explain. MESSAGES_DIR drops
      // failures into a log nobody reads from the agent's POV.
      const resultPath = scriptResultPath(sourceGroup, data);
      if (!isMain) {
        logger.warn(
          { sourceGroup },
          'Unauthorized send_message_to_chat attempt blocked',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({ error: 'send_message_to_chat is admin-tile only' }),
        );
        return;
      }

      const rawText = typeof data.text === 'string' ? data.text : '';
      if (rawText.trim().length === 0) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: 'send_message_to_chat requires non-empty text',
          }),
        );
        return;
      }

      const hasId =
        typeof data.chat_id === 'string' && data.chat_id.trim().length > 0;
      const hasName =
        typeof data.chat_name === 'string' && data.chat_name.trim().length > 0;
      if (!hasId && !hasName) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'send_message_to_chat requires chat_id or chat_name — admin always operates cross-chat. Use the regular send_message tool to reply in the current chat.',
          }),
        );
        return;
      }
      // Two identifiers are an unsafe-targeting smell — same rule as
      // chat_status / nuke_chat. Reject before resolving so the caller
      // can't get a silent JID-wins-over-name surprise.
      if (hasId && hasName) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'send_message_to_chat accepts chat_id OR chat_name, not both — they may disagree',
          }),
        );
        return;
      }

      let targetJid = '';
      if (hasId) {
        const trimmed = (data.chat_id as string).trim();
        if (!registeredGroups[trimmed]) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({ error: `chat_id ${trimmed} not registered` }),
          );
          return;
        }
        targetJid = trimmed;
      } else {
        const wanted = (data.chat_name as string).trim();
        const matches = Object.entries(registeredGroups).filter(
          ([, g]) => g.name === wanted,
        );
        if (matches.length === 0) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" did not match any registered chat`,
            }),
          );
          return;
        }
        if (matches.length > 1) {
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `chat_name "${wanted}" is ambiguous — matches ${matches.length} chats`,
              candidates: matches.map(([jid]) => jid),
            }),
          );
          return;
        }
        targetJid = matches[0][0];
      }

      const targetGroup = registeredGroups[targetJid];
      // Trim before treating as present — a payload of `'   '` would
      // otherwise route through the pool path (Boolean(' ') is true)
      // and bind a pool bot to a whitespace identity. Empty-after-trim
      // collapses to undefined so routing matches the documented
      // contract ("named identity" → pool; nothing → direct).
      const senderRaw =
        typeof data.sender === 'string' ? data.sender.trim() : '';
      const sender = senderRaw.length > 0 ? senderRaw : undefined;
      const wantsPin = data.pin === true;

      // Strip <internal> tags for parity with the regular 'message'
      // handler — keeps agent reasoning out of cross-chat broadcasts.
      // We don't apply the maintenance-prefix here: this tool is an
      // explicit admin broadcast, and tagging it `[M]` would mislead
      // the recipient into thinking a scheduled-task heartbeat fired.
      const cleanText = stripInternalTags(rawText);
      if (!cleanText) {
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error:
              'send_message_to_chat: text was empty after stripping <internal> tags',
          }),
        );
        return;
      }

      try {
        // Mirror the routing decision in the 'message' IPC handler:
        // sender + Telegram → bot pool (named identity), else the
        // channel's default sendMessage. Pool sends don't expose a
        // pin hook, so pin is silently dropped on the pool path —
        // matching existing send_message behavior. The `pinned` field
        // in the success payload tells the agent what actually
        // happened so it can flag the asymmetry to the user.
        const usePool = Boolean(sender && targetJid.startsWith('tg:'));
        let sentMsgId: string | undefined;
        let pinned = false;
        if (usePool) {
          const poolResult = await sendPoolMessage(
            targetJid,
            cleanText,
            sender!,
            sourceGroup,
          );
          sentMsgId = typeof poolResult === 'string' ? poolResult : undefined;
        } else {
          const directResult = await deps.sendMessage(targetJid, cleanText);
          sentMsgId =
            typeof directResult === 'string' ? directResult : undefined;
          if (wantsPin && sentMsgId && deps.pinMessage) {
            await deps.pinMessage(targetJid, sentMsgId);
            pinned = true;
          }
        }

        // Gate the messages.db write on send success — see
        // shouldStoreBotMessage for the phantom-row rationale (#232).
        // A failed cross-chat send that wrote a bot- row would silence
        // the target chat's heartbeat / unanswered-cron on a chat the
        // recipient never received the message in. Critical here
        // because the entire purpose of this tool is sending into
        // chats the agent isn't watching.
        if (!shouldStoreBotMessage(targetJid, sentMsgId)) {
          logger.error(
            {
              sourceGroup,
              targetJid,
              contentLen: cleanText.length,
              usedPool: usePool,
            },
            'send_message_to_chat: target returned no message id; skipping DB row',
          );
          fs.writeFileSync(
            resultPath,
            JSON.stringify({
              error: `send_message_to_chat: target ${targetJid} did not return a message id — likely blocked, rate-limited, or an invalid recipient. No DB row written.`,
              chat_id: targetJid,
              chat_name: targetGroup.name,
              status: 'failed',
            }),
          );
          return;
        }

        const botRowId = `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
        // Persist the identity that ACTUALLY went out, not what the
        // caller asked for. `sender` only takes effect on the pool
        // path (Telegram + sender set). On the direct path (non-
        // Telegram, or Telegram without sender) the message goes from
        // the channel's default identity, so storing the caller's
        // `sender` would make the DB row claim a persona that never
        // touched the wire — misleading the heartbeat / unanswered-
        // cron / future audits about who replied.
        const effectiveSender = usePool && sender ? sender : ASSISTANT_NAME;
        storeMessage({
          id: botRowId,
          chat_jid: targetJid,
          sender: effectiveSender,
          sender_name: effectiveSender,
          content: cleanText,
          timestamp: new Date().toISOString(),
          is_from_me: true,
          is_bot_message: true,
          telegram_message_id: sentMsgId,
        });

        logger.info(
          {
            sourceGroup,
            targetJid,
            sentMsgId,
            usedPool: usePool,
            pinned,
          },
          'send_message_to_chat completed',
        );
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            stdout: JSON.stringify({
              chat_id: targetJid,
              chat_name: targetGroup.name,
              sent_message_id: sentMsgId,
              pinned,
              status: 'success',
            }),
          }),
        );
        // outer-boundary-process-contract (coding-policy: error-handling):
        // IPC operation handler — the agent-runner's runHostOperation reads
        // `result.error` as the tool-failure signal.
        //   - Caller's silent-failure shape: a missing `error` field reads as
        //     success, so a swallowed failure surfaces as a phantom success.
        //   - What the catch emits: the error into `result.error` (below).
        //   - Why propagation breaks the contract: an uncaught error would
        //     skip the result envelope the caller polls for.
        // eslint-disable-next-line no-catch-all/no-catch-all -- outer-boundary-process-contract
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.error(
          { sourceGroup, targetJid, err },
          'send_message_to_chat failed',
        );
        // Top-level `error` field — runHostOperation in the
        // agent-runner only treats `result.error` as a tool failure
        // and surfaces `isError: true` to the MCP caller. Same
        // contract as nuke_chat's catch branch.
        fs.writeFileSync(
          resultPath,
          JSON.stringify({
            error: `send_message_to_chat failed for ${targetJid}: ${msg}`,
            chat_id: targetJid,
            chat_name: targetGroup.name,
            status: 'error',
          }),
        );
      }
    },
  });
}
