# NanoClaw Agent Container
# Runs Claude Agent SDK in isolated Linux VM with browser automation

FROM node:22-slim

# Install system dependencies for Chromium
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
    && rm -rf /var/lib/apt/lists/*

# Set Chromium path for agent-browser
ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium

# Install agent-browser, claude-code, and composio-mcp globally
# (tessl is not needed — tiles are delivered host-side by the orchestrator)
RUN npm install -g agent-browser @anthropic-ai/claude-code @composio/mcp

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
# Follow-up messages arrive via IPC files in /workspace/ipc/input/.
#
# Secret shadowing: Apple Container only supports directory mounts
# (VirtioFS), so the host-side /dev/null bind-mount the orchestrator
# uses on Linux (SECRET_FILES in src/container-runner.ts) can't
# shadow secret FILES under macOS Container runtime. On that runtime
# the entrypoint is the only defense. Previously this shadow covered
# only .env — missing .env.bak, data/env/env, and scripts/heartbeat-
# external.conf. Iterating over the full list closes the gap.
#
# Keep the list below in sync with SECRET_FILES in
# src/container-runner.ts. Drift would re-open exactly the hole this
# fix is closing, so if you add a path there, add it here.
RUN printf '#!/bin/bash\nset -e\n\n# Shadow secret files so agents can'"'"'t read host credentials.\n# Required on Apple Container runtime (VirtioFS rejects host-side file\n# mounts); redundant-but-harmless on Linux where the orchestrator already\n# mounts /dev/null over each secret file via SECRET_FILES.\n# Needs root + mount capability — runs before setpriv drops privileges.\nif [ "$(id -u)" = "0" ]; then\n  for secret in .env .env.bak data/env/env scripts/heartbeat-external.conf; do\n    target="/workspace/project/$secret"\n    if [ -f "$target" ]; then\n      mount --bind /dev/null "$target"\n    fi\n  done\nfi\n\n# Compile agent-runner\ncd /app && npx tsc --outDir /tmp/dist 2>&1 >&2\nln -s /app/node_modules /tmp/dist/node_modules\nchmod -R a-w /tmp/dist\n\n# Capture stdin (secrets JSON) to temp file\ncat > /tmp/input.json\n\n# Drop privileges if running as root (main-group containers)\nif [ "$(id -u)" = "0" ] && [ -n "$RUN_UID" ]; then\n  chown "$RUN_UID:$RUN_GID" /tmp/input.json /tmp/dist\n  exec setpriv --reuid="$RUN_UID" --regid="$RUN_GID" --clear-groups -- node /tmp/dist/index.js < /tmp/input.json\nfi\n\nexec node /tmp/dist/index.js < /tmp/input.json\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

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
