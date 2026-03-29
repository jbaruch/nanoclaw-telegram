# Credential Proxy Expansion Research

## Current State

The credential proxy (`src/credential-proxy.ts`, 125 lines) handles ONE service: Anthropic API. Containers send requests to `host.docker.internal:3001` with a placeholder key, the proxy swaps in the real one.

Everything else is passed as plaintext env vars directly to containers:

| Credential | Purpose | Auth Pattern |
|-----------|---------|--------------|
| `OPENAI_API_KEY` | Whisper voice transcription | `Authorization: Bearer <key>` header |
| `GITHUB_TOKEN` | GitHub API (PRs, issues) | `Authorization: token <key>` header |
| `COMPOSIO_API_KEY` | Composio MCP server | `x-consumer-api-key: <key>` header |
| `GOOGLE_CLIENT_ID` | Google Calendar OOO sync (TripIt→Reclaim) | OAuth2 — used with refresh token to get access tokens |
| `GOOGLE_CLIENT_SECRET` | Google Calendar OOO sync | OAuth2 |
| `GOOGLE_REFRESH_TOKEN` | Google Calendar OOO sync | OAuth2 — long-lived, exchanges for access token |
| `RECLAIM_API_TOKEN` | Reclaim.ai scheduling API | `Authorization: Bearer <key>` header |
| `TRIPIT_ICAL_URL` | TripIt calendar feed | URL with embedded auth (not really a secret, but specific to user) |
| Tessl OAuth | Tile registry (install/publish) | File-based OAuth tokens at `~/.tessl/api-credentials.json` |

## The Argument for Expansion

A prompt-injected agent can exfiltrate any of these. The Anthropic key is protected, but everything else is exposed. The security model is inconsistent.

## The Argument Against

- These services are WHY the agent is useful. Removing them removes capabilities.
- The proxy can't easily handle all auth patterns (API key headers, OAuth token exchange, file-based creds, URL-embedded auth).
- Latency: every API call routes through the proxy. For Anthropic (one streaming connection) it's fine. For dozens of small API calls (GitHub, Google) it adds up.
- Complexity: the proxy needs to know each service's base URL and auth injection pattern.

## Approach If We Do It

### Tier 1: Simple API key services (low effort)

Services that use a static key in a header. The proxy can match the destination host and inject.

| Service | Base URL | Header |
|---------|----------|--------|
| OpenAI | `api.openai.com` | `Authorization: Bearer <key>` |
| Reclaim | `api.reclaim.ai` | `Authorization: Bearer <key>` |
| GitHub | `api.github.com` | `Authorization: token <key>` |
| Composio | `connect.composio.dev` | `x-consumer-api-key: <key>` |

Implementation: route table in the proxy — `{ host: 'api.openai.com', header: 'Authorization', value: 'Bearer <key>' }`. Container sets `OPENAI_BASE_URL=http://host.docker.internal:3001/proxy/openai` and the proxy rewrites the host + injects auth.

~50 lines of additional proxy code. Each service needs a proxy path prefix to route correctly.

### Tier 2: Google OAuth (medium effort)

Google uses OAuth2 with refresh tokens. The agent needs an access token, which is obtained by exchanging the refresh token + client secret. Options:

**Option A:** Proxy handles token exchange. Container calls `host.docker.internal:3001/proxy/google/token` and the proxy exchanges the refresh token behind the scenes, returns an access token. The access token is short-lived (1 hour), which limits damage.

**Option B:** Pre-exchange on container start. The orchestrator gets a fresh access token before spawning the container, passes only the short-lived access token (not the refresh token or client secret). Container uses it directly. If it expires mid-session, the agent asks for a refresh via IPC.

Option B is simpler and limits exposure to a 1-hour window. The refresh token and client secret never enter the container.

### Tier 3: File-based credentials (tessl)

Already handled via read-only mount. The mount approach is fine for file-based creds — the proxy pattern doesn't apply. Could switch to a proxy if tessl supports environment-based auth in the future.

### Tier 4: URL-embedded auth (TripIt)

`TRIPIT_ICAL_URL` has a token baked into the URL. Can't easily proxy this since the URL is used by the agent's own code. Could proxy `webcal://` requests through the host, but that's overkill for a read-only calendar feed.

## Recommendation

1. **Do Tier 2 first** (Google OAuth) — it's the most sensitive credential set (client secret + refresh token = permanent access to Google Calendar). Use Option B (pre-exchange, short-lived access token only).
2. **Tier 1 second** — straightforward header injection for 4 services.
3. **Skip Tier 3 and 4** — file mount is fine for tessl, TripIt URL is low-risk.

## Decision Needed

Is the security improvement worth the complexity and latency? The primary threat model is prompt injection causing credential exfiltration. Container isolation + read-only mounts already make this hard but not impossible (the agent can still `curl` credentials out of env vars).

The nuclear option: don't pass credentials at all, make all external API calls go through IPC to the host orchestrator. The agent writes `{ type: "api_call", service: "openai", ... }` to IPC, the host makes the call, returns the result. Zero credentials in containers. But: high latency, complex IPC protocol, every new service needs host-side plumbing.
