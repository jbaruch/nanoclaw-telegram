import { describe, it, expect } from 'vitest';

import {
  AUTHORITATIVE_ENTITY_IDS,
  detectAuthoritativeLookup,
} from './authoritative-source.js';

describe('detectAuthoritativeLookup', () => {
  describe('nanoclaw-repo via composio search/list', () => {
    const matchedTools = [
      'mcp__composio__GITHUB_SEARCH_REPOSITORIES',
      'mcp__composio__github_search_issues',
      'mcp__composio__github_list_pull_requests',
      'WebSearch',
    ];
    for (const toolName of matchedTools) {
      it(`nudges when ${toolName} queries for "nanoclaw"`, () => {
        const result = detectAuthoritativeLookup(toolName, {
          query: 'nanoclaw upgrade node 25',
        });
        expect(result.nudge).toBe(true);
        expect(result.matched?.id).toBe('nanoclaw-repo');
        expect(result.systemMessage).toContain('reference_nanoclaw_repo.md');
      });
    }

    it('is silent when the search has no nanoclaw mention', () => {
      const result = detectAuthoritativeLookup(
        'mcp__composio__github_search_issues',
        { query: 'react useEffect cleanup pattern' },
      );
      expect(result.nudge).toBe(false);
    });

    it('is silent on a composio call that is not search/list', () => {
      const result = detectAuthoritativeLookup(
        'mcp__composio__github_create_issue',
        { repo: 'nanoclaw', title: 'foo' },
      );
      expect(result.nudge).toBe(false);
    });
  });

  describe('chat-jid via bash SELECT FROM chats LIMIT', () => {
    const blocked = [
      'sqlite3 messages.db "SELECT jid FROM chats LIMIT 1"',
      "sqlite3 messages.db 'SELECT jid FROM chats LIMIT 5'",
      'sqlite3 messages.db "SELECT jid, name FROM chats LIMIT 1"',
      'sqlite3 messages.db "SELECT * FROM chats ORDER BY last_message DESC LIMIT 1"',
    ];
    for (const command of blocked) {
      it(`nudges for ${JSON.stringify(command)}`, () => {
        const result = detectAuthoritativeLookup('Bash', { command });
        expect(result.nudge).toBe(true);
        expect(result.matched?.id).toBe('chat-jid');
        expect(result.systemMessage).toContain('NANOCLAW_CHAT_JID');
      });
    }

    const allowed = [
      // Scoped query — agent knows which chat it wants.
      'sqlite3 messages.db "SELECT jid FROM chats WHERE name=\'old-wtf\'"',
      // Unrelated table.
      'sqlite3 messages.db "SELECT id FROM messages LIMIT 10"',
      // Unrelated bash command.
      'echo $NANOCLAW_CHAT_JID',
    ];
    for (const command of allowed) {
      it(`is silent for ${JSON.stringify(command)}`, () => {
        expect(detectAuthoritativeLookup('Bash', { command }).nudge).toBe(
          false,
        );
      });
    }
  });

  describe('registered-groups via available_groups.json', () => {
    it('nudges when Read targets available_groups.json', () => {
      const result = detectAuthoritativeLookup('Read', {
        file_path: '/workspace/host-data/available_groups.json',
      });
      expect(result.nudge).toBe(true);
      expect(result.matched?.id).toBe('registered-groups');
      expect(result.systemMessage).toContain('registered_groups');
    });

    it('nudges when Grep searches available_groups.json', () => {
      const result = detectAuthoritativeLookup('Grep', {
        path: '/workspace/host-data/available_groups.json',
        pattern: 'old-wtf',
      });
      expect(result.nudge).toBe(true);
      expect(result.matched?.id).toBe('registered-groups');
    });

    it('nudges when Bash cats available_groups.json', () => {
      const result = detectAuthoritativeLookup('Bash', {
        command: 'jq .groups /workspace/host-data/available_groups.json',
      });
      expect(result.nudge).toBe(true);
      expect(result.matched?.id).toBe('registered-groups');
    });

    it('is silent for unrelated file reads', () => {
      const result = detectAuthoritativeLookup('Read', {
        file_path: '/workspace/group/CLAUDE.md',
      });
      expect(result.nudge).toBe(false);
    });
  });

  describe('matcher gating — cheap no-op for unrelated tools', () => {
    const unrelatedCalls: Array<[string, unknown]> = [
      [
        'Edit',
        { file_path: '/workspace/foo.ts', old_string: 'a', new_string: 'b' },
      ],
      ['Write', { file_path: '/tmp/out.txt', content: 'hi' }],
      ['mcp__nanoclaw__send_message', { chat_jid: 'tg:1', content: 'hi' }],
      ['mcp__nanoclaw__react_to_message', { id: '1', emoji: '👀' }],
      ['TaskOutput', { block: false }],
    ];
    for (const [toolName, toolInput] of unrelatedCalls) {
      it(`is silent for ${toolName}`, () => {
        expect(detectAuthoritativeLookup(toolName, toolInput).nudge).toBe(
          false,
        );
      });
    }
  });

  describe('input shape', () => {
    it('returns nudge=false for non-string tool name', () => {
      expect(detectAuthoritativeLookup(undefined, {}).nudge).toBe(false);
      expect(detectAuthoritativeLookup(null, {}).nudge).toBe(false);
      expect(detectAuthoritativeLookup(42, {}).nudge).toBe(false);
    });

    it('returns nudge=false for empty tool name', () => {
      expect(detectAuthoritativeLookup('', {}).nudge).toBe(false);
    });

    it('handles null/undefined tool input as empty', () => {
      expect(detectAuthoritativeLookup('Bash', null).nudge).toBe(false);
      expect(detectAuthoritativeLookup('Bash', undefined).nudge).toBe(false);
    });

    it('handles string tool input directly without re-serialising', () => {
      const result = detectAuthoritativeLookup(
        'WebSearch',
        'nanoclaw architecture overview',
      );
      expect(result.nudge).toBe(true);
      expect(result.matched?.id).toBe('nanoclaw-repo');
    });

    it('survives a tool input that cannot be JSON-stringified', () => {
      const circular: Record<string, unknown> = {};
      circular.self = circular;
      expect(detectAuthoritativeLookup('Bash', circular).nudge).toBe(false);
    });
  });

  it('exposes a stable entity-id catalogue', () => {
    expect(AUTHORITATIVE_ENTITY_IDS).toEqual([
      'nanoclaw-repo',
      'chat-jid',
      'registered-groups',
    ]);
  });
});
