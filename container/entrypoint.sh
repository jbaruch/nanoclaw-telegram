#!/bin/bash
set -e

# Limit git memory usage to prevent SIGBUS under container memory pressure
git config --global pack.threads 1 2>/dev/null || true
git config --global pack.deltaCacheSize 1m 2>/dev/null || true
git config --global pack.windowMemory 100m 2>/dev/null || true

# Agent-runner is pre-compiled at image build time (/app/dist/).
# No runtime compilation needed — source is mounted read-only.

# Restore pre-cached tessl native binary (build downloaded as root, runtime is uid 999)
if [ -d /opt/tessl-bin ] && [ ! -d "$HOME/.local/share/tessl/versions" ]; then
  mkdir -p "$HOME/.local/share/tessl/versions"
  cp -r /opt/tessl-bin/* "$HOME/.local/share/tessl/versions/" 2>/dev/null || true
fi

# Install tessl tiles at runtime.
# Credentials mounted read-only at /tmp/tessl-credentials.json.
# Copy to writable ~/.tessl/ (tessl writes cli.log there).
if [ -f /tmp/tessl-credentials.json ]; then
  mkdir -p "$HOME/.tessl"
  cp /tmp/tessl-credentials.json "$HOME/.tessl/api-credentials.json"

  cd /home/node/.claude
  # Ensure tessl.json exists (tessl install needs it). Empty deps is fine —
  # tile names are passed as CLI args, tessl updates the json after install.
  [ -f tessl.json ] || echo '{"name":"nanoclaw","mode":"vendored","dependencies":{}}' > tessl.json
  tessl install jbaruch/nanoclaw-core jbaruch/nanoclaw-admin jbaruch/reclaim-tripit-sync \
    --yes --dangerously-ignore-security --agent claude-code 2>&1 >&2 || echo "[entrypoint] tessl install failed" >&2

  # tessl installs to .tessl/tiles/ but SDK reads .claude/skills/.
  # Symlink each tile skill into the skills directory.
  for tile_dir in /home/node/.claude/.tessl/tiles/*/*/skills/*/; do
    [ -d "$tile_dir" ] || continue
    skill_name=$(basename "$tile_dir")
    ln -sfn "$tile_dir" "/home/node/.claude/skills/tessl__${skill_name}" 2>/dev/null || true
  done

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
node /app/dist/index.js < /tmp/input.json
