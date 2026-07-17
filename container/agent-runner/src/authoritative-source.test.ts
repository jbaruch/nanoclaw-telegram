import { describe, it, expect } from 'vitest';

import {
  AUTHORITATIVE_ENTITY_IDS,
  detectAuthoritativeLookup,
} from './authoritative-source.js';

describe('detectAuthoritativeLookup', () => {
  describe('nanoclaw-repo via WebSearch', () => {
    it('nudges when WebSearch queries for "nanoclaw"', () => {
      const result = detectAuthoritativeLookup('WebSearch', {
        query: 'nanoclaw upgrade node 25',
      });
      expect(result.nudge).toBe(true);
      expect(result.matched?.id).toBe('nanoclaw-repo');
      expect(result.systemMessage).toContain('reference_nanoclaw_repo.md');
    });

    it('is silent when the search has no nanoclaw mention', () => {
      const result = detectAuthoritativeLookup('WebSearch', {
        query: 'react useEffect cleanup pattern',
      });
      expect(result.nudge).toBe(false);
    });

    it('is silent on an MCP tool outside the nudge families', () => {
      const result = detectAuthoritativeLookup('mcp__tessl__search', {
        query: 'nanoclaw',
      });
      expect(result.nudge).toBe(false);
    });
  });

  describe('nanoclaw-repo via gh CLI over Bash (#797)', () => {
    const nudged = [
      // The documented misroute now that GitHub repo lookups run over the
      // CLI instead of Composio — nanoclaw is the ambiguous lookup target.
      'gh search repos nanoclaw',
      "gh api 'search/repositories?q=nanoclaw' --jq '.items'",
      // `gh repo view` of a bare or non-canonical-owner nanoclaw.
      'gh repo view nanoclaw',
      'gh repo view qwibitai/nanoclaw',
      // An owner that merely ends in "jbaruch" is NOT the canonical fork.
      'gh repo view notjbaruch/nanoclaw',
      // Double-quoted lookup arguments — matched against the RAW command, so
      // the quote is a plain `"`, not a JSON-escaped `\"`.
      'gh search repos "nanoclaw"',
      'gh api "search/repositories?q=nanoclaw"',
      'gh repo view "qwibitai/nanoclaw"',
    ];
    for (const command of nudged) {
      it(`nudges for ${JSON.stringify(command)}`, () => {
        const result = detectAuthoritativeLookup('Bash', { command });
        expect(result.nudge).toBe(true);
        expect(result.matched?.id).toBe('nanoclaw-repo-gh');
        expect(result.systemMessage).toContain('reference_nanoclaw_repo.md');
      });
    }

    const silent = [
      // Ordinary path-bearing commands — the false-positive class the
      // narrow inputPattern exists to avoid. Every one mentions
      // "nanoclaw" via the repo path.
      'cd ~/nanoclaw && npm test',
      'cat ~/nanoclaw/package.json',
      'ls ~/nanoclaw',
      'npm --prefix ~/nanoclaw run build',
      'git -C ~/nanoclaw status',
      'rg TODO ~/nanoclaw/src',
      // Already scoped to the right repo — a repo *action*, not a
      // fork-ambiguous lookup.
      'gh issue list --repo jbaruch/nanoclaw',
      'gh pr view 797 --repo jbaruch/nanoclaw',
      // `gh repo view` already naming the canonical fork — nothing to nudge
      // (bare and double-quoted).
      'gh repo view jbaruch/nanoclaw',
      'gh repo view "jbaruch/nanoclaw"',
      // A repo lookup of a DIFFERENT repo that merely shares a command with
      // an unrelated ~/nanoclaw path — the decoupled-token false positive,
      // across each segment separator. The matcher runs against the raw
      // `command` field (inputField: 'command'), so this is a real newline
      // (0x0A) that the segment class stops at — the lookup targets `other`,
      // and `nanoclaw` belongs to a separate command.
      'gh repo view jbaruch/other && cat ~/nanoclaw/package.json',
      'gh repo view jbaruch/other\ncat ~/nanoclaw/package.json',
      'gh repo view jbaruch/other; cat ~/nanoclaw/package.json',
      'gh repo list --limit 100 | grep nanoclaw',
      // `gh search` that is NOT a repo lookup — `gh search repos` is the
      // only nudged search shape.
      'gh search issues nanoclaw',
      'gh search code nanoclaw',
      // Hyphenated sibling repos are a DIFFERENT repo — the target is the
      // exact `nanoclaw` token, not a `nanoclaw`-prefixed name.
      'gh search repos nanoclaw-tools',
      'gh repo view qwibitai/nanoclaw-tools',
      'gh repo view someuser/mynanoclaw',
      // A github.com URL inside a Bash command must NOT nudge — the
      // github.com shape is WebFetch-only (separate entity).
      'git clone https://github.com/qwibitai/nanoclaw',
    ];
    for (const command of silent) {
      it(`is silent for ${JSON.stringify(command)}`, () => {
        expect(detectAuthoritativeLookup('Bash', { command }).nudge).toBe(
          false,
        );
      });
    }
  });

  describe('nanoclaw-repo via github.com URL over WebFetch (#797)', () => {
    it('nudges when WebFetch reads the upstream fork repo page', () => {
      const result = detectAuthoritativeLookup('WebFetch', {
        url: 'https://github.com/qwibitai/nanoclaw',
      });
      expect(result.nudge).toBe(true);
      expect(result.matched?.id).toBe('nanoclaw-repo-webfetch');
      expect(result.systemMessage).toContain('reference_nanoclaw_repo.md');
    });

    it('nudges on a deep repo-page path too', () => {
      const result = detectAuthoritativeLookup('WebFetch', {
        url: 'https://github.com/qwibitai/nanoclaw/blob/main/README.md',
      });
      expect(result.nudge).toBe(true);
      expect(result.matched?.id).toBe('nanoclaw-repo-webfetch');
    });

    it('nudges on an impersonator owner that merely ends in "jbaruch"', () => {
      const result = detectAuthoritativeLookup('WebFetch', {
        url: 'https://github.com/notjbaruch/nanoclaw',
      });
      expect(result.nudge).toBe(true);
      expect(result.matched?.id).toBe('nanoclaw-repo-webfetch');
    });

    it('nudges on a www.github.com host', () => {
      const result = detectAuthoritativeLookup('WebFetch', {
        url: 'https://www.github.com/qwibitai/nanoclaw',
      });
      expect(result.nudge).toBe(true);
      expect(result.matched?.id).toBe('nanoclaw-repo-webfetch');
    });

    const silentUrls = [
      // The canonical repo page is already the right fork — nudging there is
      // pure noise.
      'https://github.com/jbaruch/nanoclaw',
      'https://github.com/jbaruch/nanoclaw/blob/main/README.md',
      // A github.com *search* URL is not a repo page — no fork ambiguity to
      // nudge about.
      'https://github.com/search?q=nanoclaw',
      // Unrelated github.com page.
      'https://github.com/orgs/jbaruch/repositories',
      // A hyphenated sibling repo is a DIFFERENT repo, not nanoclaw.
      'https://github.com/qwibitai/nanoclaw-tools',
      // A non-github fetch that merely EMBEDS a github repo URL in its query
      // — the fetched host is example.com, so the `^…github.com` host anchor
      // must not match the embedded URL.
      'https://example.com/redacted?u=https://github.com/qwibitai/nanoclaw',
      // Non-github fetch that merely mentions the name.
      'https://example.com/nanoclaw-blog-post',
    ];
    for (const url of silentUrls) {
      it(`is silent for WebFetch ${JSON.stringify(url)}`, () => {
        expect(detectAuthoritativeLookup('WebFetch', { url }).nudge).toBe(
          false,
        );
      });
    }
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
      'nanoclaw-repo-gh',
      'nanoclaw-repo-webfetch',
      'chat-jid',
      'registered-groups',
    ]);
  });
});
