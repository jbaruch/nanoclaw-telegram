#!/bin/bash
set -euo pipefail

# Limit git memory usage to prevent SIGBUS under container memory pressure
git config --global pack.threads 1
git config --global pack.deltaCacheSize 1m
git config --global pack.windowMemory 100m

# Agent-runner is pre-compiled at image build time (/app/dist/).

# Restore pre-cached tessl native binary (build downloaded as root, runtime is uid 999)
if [ -d /opt/tessl-bin ] && [ ! -d "$HOME/.local/share/tessl/versions" ]; then
  mkdir -p "$HOME/.local/share/tessl/versions"
  cp -r /opt/tessl-bin/* "$HOME/.local/share/tessl/versions/"
fi

# Install tessl tiles at runtime.
if [ -f /tmp/tessl-credentials.json ]; then
  mkdir -p "$HOME/.tessl"
  cp /tmp/tessl-credentials.json "$HOME/.tessl/api-credentials.json"

  cd /home/node/.claude
  [ -f tessl.json ] || echo '{"name":"nanoclaw","mode":"vendored","dependencies":{}}' > tessl.json
  tessl install jbaruch/nanoclaw-core jbaruch/nanoclaw-admin jbaruch/reclaim-tripit-sync \
    --yes --dangerously-ignore-security --agent claude-code >&2

  # Symlink tile skills into .claude/skills/ where the SDK discovers them
  for tile_dir in /home/node/.claude/.tessl/tiles/*/*/skills/*/; do
    [ -d "$tile_dir" ] || continue
    skill_name=$(basename "$tile_dir")
    ln -sfn "$tile_dir" "/home/node/.claude/skills/tessl__${skill_name}"
  done

  cd /workspace/group
else
  echo "[entrypoint] WARNING: no tessl credentials at /tmp/tessl-credentials.json — tiles not installed" >&2
fi

# Copy built-in container skills (agent-browser, status, etc.) from image staging
if [ -d /opt/tessl-staging/.claude/skills ]; then
  mkdir -p /home/node/.claude/skills
  cp -rL /opt/tessl-staging/.claude/skills/* /home/node/.claude/skills/
fi

# Wire tessl rules chain into workspace
if [ -f /home/node/.claude/AGENTS.md ] && [ ! -f /workspace/group/AGENTS.md ]; then
  cp /home/node/.claude/AGENTS.md /workspace/group/AGENTS.md
fi
if [ -d /home/node/.claude/.tessl ] && [ ! -d /workspace/group/.tessl ]; then
  cp -rL /home/node/.claude/.tessl /workspace/group/.tessl
fi

# Read container input from stdin and run the agent
cat > /tmp/input.json
node /app/dist/index.js < /tmp/input.json
