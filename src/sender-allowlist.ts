import { isFileSystemError } from './operational-errors.js';
import fs from 'fs';

import { SENDER_ALLOWLIST_PATH } from './config.js';
import { logger } from './logger.js';

export interface ChatAllowlistEntry {
  allow: '*' | string[];
  mode: 'trigger' | 'drop';
}

export interface SenderAllowlistConfig {
  default: ChatAllowlistEntry;
  chats: Record<string, ChatAllowlistEntry>;
  logDenied: boolean;
}

const DEFAULT_CONFIG: SenderAllowlistConfig = {
  default: { allow: '*', mode: 'trigger' },
  chats: {},
  logDenied: true,
};

function isValidEntry(entry: unknown): entry is ChatAllowlistEntry {
  if (!entry || typeof entry !== 'object') return false;
  const e = entry as Record<string, unknown>;
  const validAllow =
    e.allow === '*' ||
    (Array.isArray(e.allow) && e.allow.every((v) => typeof v === 'string'));
  const validMode = e.mode === 'trigger' || e.mode === 'drop';
  return validAllow && validMode;
}

export function loadSenderAllowlist(
  pathOverride?: string,
): SenderAllowlistConfig {
  const filePath = pathOverride ?? SENDER_ALLOWLIST_PATH;

  let raw: string;
  try {
    raw = fs.readFileSync(filePath, 'utf-8');
  } catch (err: unknown) {
    if (!isFileSystemError(err)) throw err;
    if (err.code === 'ENOENT') return DEFAULT_CONFIG;
    throw new Error(
      'Cannot read the configured sender allowlist; repair access before processing messages.',
      { cause: err },
    );
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    if (!(err instanceof SyntaxError)) throw err;
    throw new SyntaxError(
      'Invalid sender allowlist JSON; repair the configured restriction before processing messages.',
      { cause: err },
    );
  }

  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed))
    throw new SyntaxError('Sender allowlist must be a configuration object.');
  const obj = parsed as Record<string, unknown>;

  if (!isValidEntry(obj.default))
    throw new SyntaxError(
      'Sender allowlist has an invalid default restriction; repair it before processing messages.',
    );
  if (obj.logDenied !== undefined && typeof obj.logDenied !== 'boolean')
    throw new SyntaxError('Sender allowlist logDenied must be a Boolean.');
  const chats: Record<string, ChatAllowlistEntry> = {};
  if (obj.chats !== undefined) {
    if (!obj.chats || typeof obj.chats !== 'object' || Array.isArray(obj.chats))
      throw new SyntaxError(
        'Sender allowlist chats must be a restriction map.',
      );
    for (const [jid, entry] of Object.entries(obj.chats)) {
      if (!isValidEntry(entry))
        throw new SyntaxError(
          'Sender allowlist contains an invalid chat restriction; repair it before processing messages.',
        );
      Object.defineProperty(chats, jid, { value: entry, enumerable: true });
    }
  }

  return {
    default: obj.default as ChatAllowlistEntry,
    chats,
    logDenied: obj.logDenied !== false,
  };
}

function getEntry(
  chatJid: string,
  cfg: SenderAllowlistConfig,
): ChatAllowlistEntry {
  return cfg.chats[chatJid] ?? cfg.default;
}

export function isSenderAllowed(
  chatJid: string,
  sender: string,
  cfg: SenderAllowlistConfig,
): boolean {
  const entry = getEntry(chatJid, cfg);
  if (entry.allow === '*') return true;
  return entry.allow.includes(sender);
}

export function shouldDropMessage(
  chatJid: string,
  cfg: SenderAllowlistConfig,
): boolean {
  return getEntry(chatJid, cfg).mode === 'drop';
}

export function isTriggerAllowed(
  chatJid: string,
  sender: string,
  cfg: SenderAllowlistConfig,
): boolean {
  const allowed = isSenderAllowed(chatJid, sender, cfg);
  if (!allowed && cfg.logDenied) {
    logger.debug(
      { chatJid, sender },
      'sender-allowlist: trigger denied for sender',
    );
  }
  return allowed;
}
