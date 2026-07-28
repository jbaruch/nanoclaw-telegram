# NanoClaw Agent Container
# Runs Claude Agent SDK in isolated Linux VM with browser automation

# Debian 13 (trixie) base — its system Python is 3.13, which is what the
# skills' `python3` calls resolve to inside the container. python3 is
# executed directly at runtime (e.g. container/agent-runner/src/
# ipc-mcp-stdio.ts `execSync('python3 …')`), so it's installed explicitly
# below rather than left to arrive transitively via chromium/gh — a future
# Chromium/gh dependency change must not be able to silently drop it. The
# prior `node:24-slim` tracked Debian 12 (bookworm) → Python 3.11.
#
# Pinned by digest, not just the floating `24-trixie-slim` tag, so builds
# are reproducible and the Debian layer under Node only advances through
# a reviewed bump (`coding-policy: dependency-management` Pinning). The
# digest is kept fresh by the `docker` ecosystem in `.github/dependabot.yml`,
# which opens a PR when the tag is re-published; the tag stays in the ref
# as the human-readable channel, the digest is authoritative.
FROM node:26-trixie-slim@sha256:715e55e4b84e4bb0ff48e49b398a848f08e55daed8eb6a0ea1839ae53bc57583

# Install system dependencies for Chromium + GitHub CLI.
# `gh` ships from GitHub's official apt repo (not in default debian-slim
# sources). Used inside main/trusted-tier containers by the cost-monitor
# dashboard skills (precheck-gating-monitor, session-cap-monitor,
# daily-spend-rollup) to edit dashboard issues and post anomaly comments
# directly via `gh issue edit/comment`. CLI authenticates from
# GITHUB_TOKEN automatically (orchestrator
# injects via --env-file per SECRET_CONTAINER_VARS in
# src/container-runner.ts).
RUN apt-get update && apt-get install -y \
    chromium \
    fonts-liberation \
    fonts-noto-color-emoji \
    libgbm1 \
    libnss3 \
    libatk-bridge2.0-0 \
    libgtk-3-0 \
    libx11-xcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libasound2 \
    libpangocairo-1.0-0 \
    libcups2 \
    libdrm2 \
    libxshmfence1 \
    curl \
    git \
    poppler-utils \
    python3 \
    sqlite3 \
 && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | tee /usr/share/keyrings/githubcli-archive-keyring.gpg > /dev/null \
 && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
 && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list \
 && apt-get update \
 && apt-get install -y gh \
 && rm -rf /var/lib/apt/lists/*

# Set Chromium path for agent-browser and default screenshots to JPEG.
# agent-browser falls back to PNG without these knobs, which bloats cached
# prompts during multi-screenshot browser flows. JPEG quality 75 preserves
# enough visual detail for agent pattern matching while cutting image tokens.
ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium
ENV AGENT_BROWSER_SCREENSHOT_FORMAT=jpeg
ENV AGENT_BROWSER_SCREENSHOT_QUALITY=75

# Install agent-browser and claude-code globally
# (tessl is not needed — tiles are delivered host-side by the orchestrator)
#
# Pinned so two rebuilds of the same commit produce the same image
# (`coding-policy: dependency-management` Pinning). Unpinned, the agent's
# own runtime could change under a rebuild that carries no diff — including
# a `deploy.sh --no-cache` run whose stated purpose was something else.
# Renewal is a committed scanner, not memory: `.github/renovate.json`'s
# `custom.regex` manager watches these two literals and opens a bump PR, so
# a version change is reviewed rather than silently inherited (#796).
RUN npm install -g agent-browser@0.32.1 @anthropic-ai/claude-code@2.1.212

# #748: the TripIt → Reclaim timezone sync moves in-container (creds swapped at
# the OneCLI gateway), so the agent runs `reclaim-tripit-timezones-sync` itself
# instead of the host-op that ran it in the orchestrator (removed in the #748
# Phase 3 cutover, once the in-container path was verified).
# Deliberately NOT pinned — approved exception under `coding-policy:
# dependency-management` First-Party Co-Shipped Dependency Carve-Out (0.3.129),
# with `nanoclaw-host: sync-cli-floating` (0.1.54) as the authority-of-record.
# Same operator owns both repos and this Dockerfile is the package's only
# consumer, so a pin never stood between us and unreviewed upstream code; what
# it did was hold the image on a build from before the package grew the OneCLI
# support this container needs. That cost 19 days of the nightly sync creating
# zero OOO blocks: `createGCalClient` still wanted the GOOGLE_* OAuth trio that #638
# removed, returned null, and the CLI read that as "OOO not configured" — while
# Renovate, watching github-tags, saw no newer tag to propose because none was
# ever cut. The package's own CI on `main` is what replaces the bump review.
#
# The ADD below is load-bearing, not decoration. Without it BuildKit caches this
# layer forever and "unpinned" quietly means "whatever the last rebuild
# fetched" — a pin with no version anyone can read. The commit JSON changes
# whenever upstream `main` moves, which invalidates the layer and forces the
# reinstall. `deploy.sh` step 3b-ter fails the deploy if either half drifts.
# Kept on its own layer.
ADD https://api.github.com/repos/jbaruch/reclaim-tripit-timezones-sync/commits/main /tmp/rtts-head.json
RUN npm install -g jbaruch/reclaim-tripit-timezones-sync && rm -f /tmp/rtts-head.json

# Create app directory
WORKDIR /app

# Copy package files first for better caching
COPY agent-runner/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY agent-runner/ ./

# Build TypeScript
RUN npm run build

# Create workspace directories
RUN mkdir -p /workspace/group /workspace/global /workspace/extra /workspace/ipc/messages /workspace/ipc/tasks /workspace/ipc/input

# Ensure .claude directory exists
RUN mkdir -p /home/node/.claude

# Create entrypoint script
# Container input (prompt, group info) is passed via stdin JSON.
# Credentials are injected by the host's credential proxy — never passed here.
# Follow-up messages arrive via IPC files in /workspace/ipc/input/
# Apple Container only supports directory mounts (VirtioFS), so .env cannot be
# shadowed with a host-side /dev/null file mount. Instead the entrypoint starts
# as root, uses mount --bind to shadow .env, then drops to the host user via setpriv.
RUN printf '#!/bin/bash\nset -e\n\n# Shadow .env so the agent cannot read host secrets (requires root)\nif [ "$(id -u)" = "0" ] && [ -f /workspace/project/.env ]; then\n  mount --bind /dev/null /workspace/project/.env\nfi\n\n# Compile agent-runner\ncd /app && npx tsc --outDir /tmp/dist 2>&1 >&2\nln -s /app/node_modules /tmp/dist/node_modules\nchmod -R a-w /tmp/dist\n\n# Capture stdin (secrets JSON) to temp file\ncat > /tmp/input.json\n\n# Drop privileges if running as root (main-group containers)\nif [ "$(id -u)" = "0" ] && [ -n "$RUN_UID" ]; then\n  chown "$RUN_UID:$RUN_GID" /tmp/input.json /tmp/dist\n  exec setpriv --reuid="$RUN_UID" --regid="$RUN_GID" --clear-groups -- node /tmp/dist/index.js < /tmp/input.json\nfi\n\nexec node /tmp/dist/index.js < /tmp/input.json\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set ownership to node user (non-root) for writable directories
RUN chown -R node:node /workspace && \
    chmod 777 /home/node

# Limit git memory usage (baked in so read-only rootfs containers don't fail)
RUN git config --global pack.threads 1 && \
    git config --global pack.deltaCacheSize 1m && \
    git config --global pack.windowMemory 100m


# Run as non-root node user (uid 1000) by default.
# The orchestrator may override with --user for non-main groups.
USER node

# Set working directory to group workspace
WORKDIR /workspace/group

# Entry point reads JSON from stdin, outputs JSON to stdout
ENTRYPOINT ["/app/entrypoint.sh"]
