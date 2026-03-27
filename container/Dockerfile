# NanoClaw Agent Container
# Runs Claude Agent SDK in isolated Linux VM with browser automation

FROM node:22-slim

# Install system dependencies for Chromium
RUN apt-get update && apt-get install -y \
    chromium \
    fonts-liberation \
    fonts-noto-cjk \
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
    && rm -rf /var/lib/apt/lists/*

# Set Chromium path for agent-browser
ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium

# Install agent-browser, claude-code, tessl, and composio-mcp globally
RUN npm install -g agent-browser @anthropic-ai/claude-code tessl @composio/mcp

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

# Install tessl tiles into a staging directory.
# At runtime, /home/node/.claude is bind-mounted (masking image contents).
# The entrypoint copies staged tiles into the bind mount on startup.
WORKDIR /opt/tessl-staging
# Pre-download the tessl native binary, then install tiles.
# Write API key from secret into tessl credentials for private tile access.
# The credentials file is overwritten after install to remove the key from the layer.
RUN tessl --version 2>/dev/null || true
RUN echo '{"name":"nanoclaw","mode":"vendored","dependencies":{}}' > tessl.json \
    && mkdir -p .claude/skills
RUN --mount=type=secret,id=tessl_key \
    mkdir -p "$HOME/.tessl" && \
    printf '{"accessToken":"%s"}' "$(cat /run/secrets/tessl_key)" > "$HOME/.tessl/api-credentials.json" && \
    tessl install jbaruch/nanoclaw-core jbaruch/nanoclaw-admin --yes --dangerously-ignore-security --agent claude-code && \
    rm -f "$HOME/.tessl/api-credentials.json" && \
    echo "Installed tiles:" && find . -name 'SKILL.md' -o -name 'RULES.md' | head -20

# Mirror macOS Google Drive path so symlinks like ~/.claude/blog-writer-persona resolve in container
# Host: /Users/jbaruch/Library/CloudStorage/GoogleDrive-.../My Drive/Blogs → /workspace/extra/blogs
RUN mkdir -p "/Users/jbaruch/Library/CloudStorage/GoogleDrive-jbaruch@sadogursky.com/My Drive" \
    && ln -s /workspace/extra/blogs    "/Users/jbaruch/Library/CloudStorage/GoogleDrive-jbaruch@sadogursky.com/My Drive/Blogs" \
    && ln -s /workspace/extra/presentations "/Users/jbaruch/Library/CloudStorage/GoogleDrive-jbaruch@sadogursky.com/My Drive/Presentations" \
    && ln -s /workspace/extra/travel   "/Users/jbaruch/Library/CloudStorage/GoogleDrive-jbaruch@sadogursky.com/My Drive/Travel" \
    && mkdir -p /home/node/.claude \
    && ln -s /workspace/extra/blogs/persona /home/node/.claude/blog-writer-persona

WORKDIR /app

# Create entrypoint script
# 1. Compile agent-runner from (potentially customized) /app/src
# 2. Copy tessl-staged tiles into the bind-mounted .claude/ directory
# 3. Read container input from stdin and run the agent
RUN printf '#!/bin/bash\nset -e\n\n# Compile agent-runner\ncd /app && npx tsc --outDir /tmp/dist 2>&1 >&2\nln -s /app/node_modules /tmp/dist/node_modules\nchmod -R a-w /tmp/dist\n\n# Copy staged tessl tiles into bind-mounted .claude/\nif [ -d /opt/tessl-staging/.claude/skills ]; then\n  cp -rn /opt/tessl-staging/.claude/skills/* /home/node/.claude/skills/ 2>/dev/null || true\nfi\nif [ -d /opt/tessl-staging/.tessl ]; then\n  cp -rn /opt/tessl-staging/.tessl /home/node/.tessl-tiles 2>/dev/null || true\nfi\n# Merge AGENTS.md (tessl rules chain) if not already present\nif [ -f /opt/tessl-staging/AGENTS.md ] && [ ! -f /workspace/group/AGENTS.md ]; then\n  cp /opt/tessl-staging/AGENTS.md /workspace/group/AGENTS.md 2>/dev/null || true\nfi\nif [ -f /opt/tessl-staging/CLAUDE.md ] && ! grep -q "AGENTS.md" /workspace/group/CLAUDE.md 2>/dev/null; then\n  echo "" >> /workspace/group/CLAUDE.md\n  cat /opt/tessl-staging/CLAUDE.md >> /workspace/group/CLAUDE.md 2>/dev/null || true\nfi\nif [ -d /opt/tessl-staging/.tessl ] && [ ! -d /workspace/group/.tessl ]; then\n  cp -r /opt/tessl-staging/.tessl /workspace/group/.tessl 2>/dev/null || true\nfi\n\ncat > /tmp/input.json\nnode /tmp/dist/index.js < /tmp/input.json\n' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set ownership to node user (non-root) for writable directories
RUN chown -R node:node /workspace /opt/tessl-staging && chmod 777 /home/node

# Switch to non-root user (required for --dangerously-skip-permissions)
USER node

# Set working directory to group workspace
WORKDIR /workspace/group

# Entry point reads JSON from stdin, outputs JSON to stdout
ENTRYPOINT ["/app/entrypoint.sh"]
