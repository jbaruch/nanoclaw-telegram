#!/bin/bash
set -e

# Limit git memory usage to prevent SIGBUS under container memory pressure
git config --global pack.threads 1 2>/dev/null || true
git config --global pack.deltaCacheSize 1m 2>/dev/null || true
git config --global pack.windowMemory 100m 2>/dev/null || true

# Compile agent-runner from (potentially customized) /app/src
cd /app && npx tsc --outDir /tmp/dist 2>&1 >&2
ln -s /app/node_modules /tmp/dist/node_modules
chmod -R a-w /tmp/dist

# Install tessl tiles at runtime (credentials mounted read-only from host)
# This replaces the build-time staging approach — always gets latest tile versions.
if [ -f /home/node/.tessl/api-credentials.json ]; then
  cd /home/node/.claude
  echo '{"name":"nanoclaw","mode":"vendored","dependencies":{}}' > tessl.json 2>/dev/null || true
  tessl install jbaruch/nanoclaw-core jbaruch/nanoclaw-admin jbaruch/reclaim-tripit-sync \
    --yes --dangerously-ignore-security --agent claude-code 2>&1 | head -5 >&2 || true
  cd /workspace/group
fi

# Copy built-in container skills (agent-browser, status, etc.) from image staging
if [ -d /opt/tessl-staging/.claude/skills ]; then
  mkdir -p /home/node/.claude/skills
  cp -rL /opt/tessl-staging/.claude/skills/* /home/node/.claude/skills/ 2>/dev/null || true
fi

# Wire tessl rules chain into workspace (CLAUDE.md → AGENTS.md → .tessl/RULES.md)
if [ -f /home/node/.claude/AGENTS.md ] && [ ! -f /workspace/group/AGENTS.md ]; then
  cp /home/node/.claude/AGENTS.md /workspace/group/AGENTS.md 2>/dev/null || true
fi
if [ -d /home/node/.claude/.tessl ] && [ ! -d /workspace/group/.tessl ]; then
  cp -rL /home/node/.claude/.tessl /workspace/group/.tessl 2>/dev/null || true
fi

# Read container input from stdin and run the agent
cat > /tmp/input.json
node /tmp/dist/index.js < /tmp/input.json
