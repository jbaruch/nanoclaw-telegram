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
    poppler-utils \
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

# Stage built-in container skills (agent-browser, status, etc.)
# These are not tessl tiles — they ship with the image.
# Tessl tiles are installed at runtime by the entrypoint.
WORKDIR /opt/tessl-staging
RUN mkdir -p .claude/skills
COPY skills/ /opt/tessl-staging/.claude/skills/

# Pre-download the tessl native binary to a shared location.
# tessl caches to ~/.local/share/tessl/versions/ — but at runtime the container
# runs as uid 999 (not root), so we download as root then move to a shared path.
RUN tessl --version 2>/dev/null || true; \
    mkdir -p /opt/tessl-bin && \
    cp -r /root/.local/share/tessl/versions/* /opt/tessl-bin/ 2>/dev/null || true && \
    chmod -R a+rX /opt/tessl-bin

# Ensure .claude directory exists for tile staging copy at runtime
RUN mkdir -p /home/node/.claude

WORKDIR /app

# Copy entrypoint script (compiles agent-runner, copies tessl tiles, runs agent)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set ownership to node user (non-root) for writable directories
RUN chown -R node:node /workspace /opt/tessl-staging && \
    chmod -R a+rX /opt/tessl-staging && \
    chmod 777 /home/node

# Switch to non-root user (required for --dangerously-skip-permissions)
USER node

# Set working directory to group workspace
WORKDIR /workspace/group

# Entry point reads JSON from stdin, outputs JSON to stdout
ENTRYPOINT ["/app/entrypoint.sh"]
