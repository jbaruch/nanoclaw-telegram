import { describe, it, expect } from 'vitest';

import { byairMcpServer } from './byair-mcp.js';

// #645 — the agent-runner spreads byairMcpServer(process.env.BYAIR_MCP_URL)
// into mcpServersConfig. These tests pin the registration shape so a
// regression can't silently drop the byair tool surface or register it
// with the wrong transport.

describe('byairMcpServer (#645)', () => {
  it('returns {} when the URL is undefined (no-op spread)', () => {
    expect(byairMcpServer(undefined)).toEqual({});
  });

  it('returns {} when the URL is an empty string', () => {
    expect(byairMcpServer('')).toEqual({});
  });

  it('registers an http MCP server with the URL verbatim when set', () => {
    const url = 'https://api.byairapp.com/mcp?api_key=byair_test123';
    expect(byairMcpServer(url)).toEqual({
      byair: { type: 'http', url },
    });
  });

  it('does not strip or rewrite the inline api_key query param', () => {
    const url = 'https://api.byairapp.com/mcp?api_key=byair_secret&foo=bar';
    expect(byairMcpServer(url).byair.url).toBe(url);
  });
});
