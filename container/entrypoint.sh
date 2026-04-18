#!/bin/bash
set -euo pipefail

# Git memory limits are baked into the image (Dockerfile).
# Agent-runner is pre-compiled at image build time (/app/dist/).

# Secret shadowing is ORCHESTRATOR-SIDE, not here. See `SECRET_FILES`
# in src/container-runner.ts: for main groups the orchestrator
# bind-mounts /dev/null over each secret file when spawning the
# container, giving the agent a zero-byte file where a bot token
# would otherwise live. Non-main groups don't mount
# /workspace/project/ at all, so their containers never see these
# files in the first place.
#
# The previous in-container `mount --bind /dev/null ...` loop was
# dead code in both paths: for non-main groups the project dir
# didn't exist, and for main groups the mount syscall fails under
# normal container capabilities (needs CAP_SYS_ADMIN), with a
# `2>/dev/null || true` suppressing the failure silently. Real
# shadowing was always happening host-side; the in-container loop
# gave a false sense of defense in depth AND diverged from the
# canonical SECRET_FILES list (missing .env.bak and
# scripts/heartbeat-external.conf).
#
# Adding a new secret file? Extend SECRET_FILES in
# container-runner.ts — single source of truth.

# Wire tessl rules chain into workspace (first-time setup for new groups).
# .tessl/ and skills/ are populated host-side by container-runner.
# May fail on read-only filesystems (untrusted groups) — non-fatal.
if [ -w /workspace/group ]; then
  if [ -d /home/node/.claude/.tessl ] && [ ! -d /workspace/group/.tessl ]; then
    cp -rL /home/node/.claude/.tessl /workspace/group/.tessl
    echo "[entrypoint] Copied .tessl to workspace" >&2
  fi
  if [ -f /home/node/.claude/.tessl/RULES.md ] && [ ! -f /workspace/group/AGENTS.md ]; then
    cat > /workspace/group/AGENTS.md << 'AGENTS_EOF'


# Agent Rules <!-- managed by orchestrator -->

@.tessl/RULES.md follow the [instructions](.tessl/RULES.md)
AGENTS_EOF
    echo "[entrypoint] Created AGENTS.md" >&2
  fi
fi

# Read container input from stdin and run the agent
cat > /tmp/input.json
node /app/dist/index.js < /tmp/input.json
