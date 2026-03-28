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

# Copy tessl-staged tiles into bind-mounted .claude/
# Use -L to dereference symlinks (tessl install creates symlinks to vendored tiles)
# First remove any broken symlinks from previous runs (they block -n no-clobber)
if [ -d /opt/tessl-staging/.claude/skills ]; then
  mkdir -p /home/node/.claude/skills
  find /home/node/.claude/skills -maxdepth 1 -type l ! -exec test -e {} \; -delete 2>/dev/null || true
  cp -rL /opt/tessl-staging/.claude/skills/* /home/node/.claude/skills/ 2>/dev/null || true
fi

# Copy tessl tile data (rules, docs) for reference
if [ -d /opt/tessl-staging/.tessl ]; then
  cp -rLn /opt/tessl-staging/.tessl /home/node/.tessl-tiles 2>/dev/null || true
fi

# Wire tessl rules chain into workspace (CLAUDE.md → AGENTS.md → .tessl/RULES.md)
if [ -f /opt/tessl-staging/AGENTS.md ] && [ ! -f /workspace/group/AGENTS.md ]; then
  cp /opt/tessl-staging/AGENTS.md /workspace/group/AGENTS.md 2>/dev/null || true
fi
if [ -f /opt/tessl-staging/CLAUDE.md ] && ! grep -q "AGENTS.md" /workspace/group/CLAUDE.md 2>/dev/null; then
  echo "" >> /workspace/group/CLAUDE.md
  cat /opt/tessl-staging/CLAUDE.md >> /workspace/group/CLAUDE.md 2>/dev/null || true
fi
if [ -d /opt/tessl-staging/.tessl ] && [ ! -d /workspace/group/.tessl ]; then
  cp -rL /opt/tessl-staging/.tessl /workspace/group/.tessl 2>/dev/null || true
fi

# Read container input from stdin and run the agent
cat > /tmp/input.json
node /tmp/dist/index.js < /tmp/input.json
