import fs from 'fs';
import path from 'path';

import { ASSISTANT_NAME, GROUPS_DIR } from '../config.js';

import { sendPoolMessage } from '../channels/telegram.js';
import { shouldStoreBotMessage, storeMessage } from '../db-messages.js';
import {
  registerIpcMessageHandler,
  type IpcMessageContext,
} from '../ipc-message-registry.js';
import { logger } from '../logger.js';
import { stripInternalTags } from '../router.js';
import { applyMaintenancePrefix } from '../maintenance-prefix.js';

/**
 * The outbound-message IPC surface (#878): `react_to_message`,
 * `send_file`, and `message` (the wire type the `send_message` MCP tool
 * emits).
 *
 * #845 put the TASK surface on a registry but left these three inline in
 * `startIpcWatcher`'s messages loop, so OCP was complete for tasks only.
 * The bodies here are verbatim transplants from that loop (modulo
 * indentation); each keeps its original authorization gate — "main, OR
 * the target chat belongs to the sending group", evaluated on the
 * directory-verified `sourceGroup`/`isMain`, never on payload fields.
 *
 * Two guard shapes moved out of the old `else if` conditions and into
 * each handler's early return, because dispatch now keys on `data.type`
 * alone:
 *   - the required payload fields (`chatJid`, `emoji`, `filePath`, `text`)
 *   - the optional channel deps (`sendReaction`, `sendFile`) that a given
 *     deployment may not provide
 * A failed guard returns without acting and the poller deletes the file —
 * identical to pre-#878, where such a payload matched no branch and fell
 * through to the same unlink (there was never a trailing `else`).
 */
async function reactToMessage(ctx: IpcMessageContext): Promise<void> {
  const { data, sourceGroup, isMain, registeredGroups, deps } = ctx;
  const { chatJid, emoji } = data;
  const sendReaction = deps.sendReaction;
  if (!chatJid || !emoji || !sendReaction) return;
  const targetGroup = registeredGroups[chatJid];
  if (isMain || (targetGroup && targetGroup.folder === sourceGroup)) {
    await sendReaction(chatJid, data.messageId || undefined, emoji);
    logger.info(
      {
        chatJid: chatJid,
        emoji: emoji,
        sourceGroup,
      },
      'IPC reaction sent',
    );
  } else {
    logger.warn(
      { chatJid: chatJid, sourceGroup },
      'Unauthorized IPC reaction attempt blocked',
    );
  }
}

async function sendFile(ctx: IpcMessageContext): Promise<void> {
  const { data, sourceGroup, isMain, registeredGroups, deps } = ctx;
  const { chatJid } = data;
  const payloadFilePath = data.filePath;
  const sendFileDep = deps.sendFile;
  if (!chatJid || !payloadFilePath || !sendFileDep) return;
  const targetGroup = registeredGroups[chatJid];
  if (isMain || (targetGroup && targetGroup.folder === sourceGroup)) {
    // Translate container path to host path
    const containerPath: string = payloadFilePath;
    let hostPath: string;
    if (containerPath.startsWith('/workspace/group/')) {
      hostPath = path.join(
        GROUPS_DIR,
        sourceGroup,
        containerPath.replace('/workspace/group/', ''),
      );
    } else if (containerPath.startsWith('/workspace/trusted/')) {
      hostPath = path.join(
        process.cwd(),
        'trusted',
        containerPath.replace('/workspace/trusted/', ''),
      );
    } else {
      logger.warn(
        { containerPath, sourceGroup },
        'send_file: path outside allowed mounts',
      );
      // Pre-#878 this unlinked the IPC file and `continue`d the poller
      // loop. The poller now unlinks after every dispatch, so a bare
      // return deletes it exactly once — unlinking here too would make
      // the poller's own unlink throw ENOENT and quarantine a file that
      // no longer exists.
      return;
    }

    if (fs.existsSync(hostPath)) {
      // Strip <internal>…</internal> blocks from the caption
      // so agent-written internal reasoning never leaks —
      // neither to Telegram (display) nor to messages.db
      // (which feeds heartbeat's answered-check accounting).
      // Mirrors the message-payload stripping below. If the
      // caption is fully internal, send the file with no
      // caption; the file itself is still useful payload.
      const strippedCaption = data.caption
        ? stripInternalTags(data.caption)
        : '';
      // Tag maintenance-session captions so Baruch can
      // tell a scheduled-task file-send from a live one.
      // Skip the prefix entirely when the caption is
      // empty — `[M] ` alone on a silent file-send is
      // noise.
      const cleanCaption = strippedCaption
        ? applyMaintenancePrefix(
            strippedCaption,
            typeof data.sessionName === 'string' ? data.sessionName : undefined,
          )
        : '';
      const sentFileMsgId = await sendFileDep(
        chatJid,
        hostPath,
        cleanCaption || undefined,
        data.replyToMessageId,
      );
      // Store the cleaned caption (if any) so the message
      // shows up in accounting the same as text messages.
      // Without this, `send_file` is a bypass: captions
      // reach Telegram but never hit messages.db, so
      // heartbeat unanswered-checks think the agent never
      // responded. Store the cleaned version — storing the
      // raw caption would let a caption whose visible text
      // was empty after stripping count as an "answered"
      // response. Gate on `sentFileMsgId` (#428) — a
      // failed send must not leave a phantom row that
      // marks the user as answered when delivery never
      // landed.
      const captionDelivered = shouldStoreBotMessage(chatJid, sentFileMsgId);
      if (cleanCaption && captionDelivered) {
        storeMessage({
          id: `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
          chat_jid: chatJid,
          sender: ASSISTANT_NAME,
          sender_name: ASSISTANT_NAME,
          content: cleanCaption,
          timestamp: new Date().toISOString(),
          is_from_me: true,
          is_bot_message: true,
          reply_to_message_id: data.replyToMessageId,
          // Stamp the Telegram message id so post-hoc
          // "which bot send corresponds to Telegram
          // message X" queries match the orchestrator
          // text-reply path (`src/index.ts:1659`) and
          // the `send_message` handler. Pre-#428
          // sendFile returned void so this column was
          // unavailable; now that we have the id, no
          // reason not to record it.
          telegram_message_id: sentFileMsgId,
        });
      } else if (cleanCaption && !captionDelivered) {
        logger.warn(
          {
            chatJid: chatJid,
            hostPath,
            captionLen: cleanCaption.length,
          },
          'send_file: skipping caption storeMessage — sendFile returned no message id (delivery failed)',
        );
      }
      // #722: same visible-send anchor release as the
      // send_message path — a delivered file (with or
      // without caption) is a visible reply.
      if (typeof sentFileMsgId === 'string') {
        deps.onVisibleReply?.(chatJid, sourceGroup);
      }
      logger.info({ chatJid: chatJid, hostPath, sourceGroup }, 'IPC file sent');
    } else {
      logger.warn(
        { hostPath, containerPath, sourceGroup },
        'send_file: file not found on host',
      );
    }
  }
}

async function sendMessage(ctx: IpcMessageContext): Promise<void> {
  const { data, sourceGroup, isMain, registeredGroups, deps, file } = ctx;
  const { chatJid, text } = data;
  if (!chatJid || !text) return;
  logger.debug(
    {
      sourceGroup,
      chatJid: chatJid,
      rawTextLen: text.length,
      rawPreview: String(text).slice(0, 80),
      hasSender: Boolean(data.sender),
      senderValue: data.sender,
      hasReplyTo: Boolean(data.replyToMessageId),
      hasPin: Boolean(data.pin),
      ipcFile: file,
    },
    '[ipc] Received send_message IPC',
  );
  // Strip <internal> tags via the shared helper so this
  // path can't drift from the send_file caption path
  // above. If nothing remains, skip silently.
  const strippedText = stripInternalTags(text);
  if (!strippedText) {
    logger.debug(
      { sourceGroup },
      '[ipc] send_message suppressed (all internal)',
    );
    // Pre-#878 this unlinked the IPC file and `continue`d the poller
    // loop. The poller now unlinks after every dispatch, so a bare
    // return deletes it exactly once — unlinking here too would make
    // the poller's own unlink throw ENOENT and quarantine a file that
    // no longer exists.
    return;
  }
  // Tag maintenance-session text so Baruch can tell a
  // scheduled-task reply from a live conversational one.
  // Applied AFTER internal-tag stripping (no point
  // prefixing text we're about to suppress) and BEFORE
  // both the Telegram send and the messages.db store, so
  // the prefix flows through accounting uniformly.
  const cleanText = applyMaintenancePrefix(
    strippedText,
    typeof data.sessionName === 'string' ? data.sessionName : undefined,
  );
  logger.debug(
    {
      sourceGroup,
      chatJid: chatJid,
      cleanLen: cleanText.length,
      cleanPreview: cleanText.slice(0, 80),
    },
    '[ipc] send_message after stripInternalTags + maintenance-prefix',
  );

  // Authorization: verify this group can send to this chatJid
  const targetGroup = registeredGroups[chatJid];
  const authOk =
    isMain || Boolean(targetGroup && targetGroup.folder === sourceGroup);
  logger.debug(
    {
      sourceGroup,
      chatJid: chatJid,
      isMain,
      targetGroupFolder: targetGroup?.folder,
      authOk,
    },
    '[ipc] send_message auth check',
  );
  if (authOk) {
    const usePool = Boolean(data.sender && chatJid.startsWith('tg:'));
    logger.debug(
      {
        sourceGroup,
        chatJid: chatJid,
        path: usePool ? 'pool' : 'direct',
        sender: data.sender,
      },
      '[ipc] send_message path decision',
    );
    // Capture whichever send path's message ID applies. Both
    // `sendPoolMessage` and `deps.sendMessage` return the
    // Telegram-native message ID (or undefined if the send
    // failed or the channel isn't Telegram). Stored on the
    // messages row so "which bot send produced Telegram ID X"
    // is queryable without log spelunking.
    // Normalize immediately: both send paths can return
    // `string | void | undefined`. Collapsing to the
    // `string | undefined` domain up front keeps downstream
    // uses (`pinMessage`, `storeMessage`) type-safe without
    // truthiness checks that would also drop legitimate
    // empty-string / '0' IDs if Telegram ever returns them.
    let sentMsgId: string | undefined;
    if (usePool) {
      // `usePool` is only true when `data.sender` is a non-
      // empty string — TS just can't re-narrow across the
      // intermediate `Boolean(...)` boundary. The `!` is
      // safe by the `usePool` definition directly above.
      const poolResult = await sendPoolMessage(
        chatJid,
        cleanText,
        data.sender!,
        sourceGroup,
      );
      sentMsgId = typeof poolResult === 'string' ? poolResult : undefined;
      logger.debug(
        {
          sourceGroup,
          chatJid: chatJid,
          sentMsgId,
        },
        '[ipc] sendPoolMessage returned',
      );
    } else {
      const directResult = await deps.sendMessage(
        chatJid,
        cleanText,
        data.replyToMessageId,
      );
      sentMsgId = typeof directResult === 'string' ? directResult : undefined;
      logger.debug(
        {
          sourceGroup,
          chatJid: chatJid,
          sentMsgId,
        },
        '[ipc] deps.sendMessage returned',
      );
      // Pin the message if requested
      if (data.pin && sentMsgId && deps.pinMessage) {
        await deps.pinMessage(chatJid, sentMsgId);
        logger.debug(
          { sourceGroup, chatJid: chatJid, sentMsgId },
          '[ipc] pinMessage returned',
        );
      }
    }
    // #722: a confirmed visible reply releases the target
    // chat's reply anchor at the send boundary — the SDK
    // result's mark-displayed consumption arrives too late
    // for pipes landing in the gap. Own-chat gating lives
    // in the consumer (src/index.ts).
    if (typeof sentMsgId === 'string') {
      deps.onVisibleReply?.(chatJid, sourceGroup);
    }
    // Gate the bot-row write on send success — see the
    // `shouldStoreBotMessage` helper for the full rationale
    // (phantom rows on swallowed Telegram sends would
    // silence the heartbeat / unanswered-cron and feed
    // cascading hallucinated quote-replies downstream).
    if (shouldStoreBotMessage(chatJid, sentMsgId)) {
      const botRowId = `bot-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      storeMessage({
        id: botRowId,
        chat_jid: chatJid,
        sender: data.sender || ASSISTANT_NAME,
        sender_name: data.sender || ASSISTANT_NAME,
        content: cleanText,
        timestamp: new Date().toISOString(),
        is_from_me: true,
        is_bot_message: true,
        reply_to_message_id: data.replyToMessageId,
        telegram_message_id: sentMsgId,
      });
      logger.info(
        {
          chatJid: chatJid,
          sourceGroup,
          botRowId,
          contentLen: cleanText.length,
        },
        '[ipc] send_message complete — DB row written',
      );
    } else {
      logger.error(
        {
          chatJid: chatJid,
          sourceGroup,
          contentLen: cleanText.length,
        },
        '[ipc] send_message failed — Telegram returned no message id; skipping DB row to avoid phantom bot reply (would silence heartbeat / unanswered alerts)',
      );
    }
  } else {
    logger.warn(
      {
        chatJid: chatJid,
        sourceGroup,
        targetGroupFolder: targetGroup?.folder,
      },
      '[ipc] Unauthorized IPC message attempt blocked',
    );
  }
}

let registered = false;

/**
 * Register the three outbound-message commands. Idempotent so both
 * `startIpcWatcher` and direct test callers can invoke it without a
 * duplicate-registration throw.
 */
export function registerMessageIpcHandlers(): void {
  if (registered) return;
  registered = true;
  registerIpcMessageHandler('react_to_message', reactToMessage);
  registerIpcMessageHandler('send_file', sendFile);
  registerIpcMessageHandler('message', sendMessage);
}

/**
 * Reset the once-guard alongside `_resetIpcMessageRegistryForTests`.
 *
 * @internal — test-only export, stripped from the public `.d.ts`
 * surface (`stripInternal: true`).
 */
export function _resetMessageIpcHandlersForTests(): void {
  registered = false;
}
