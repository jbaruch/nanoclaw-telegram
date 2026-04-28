#!/usr/bin/env bash
#
# check-checkpoint.sh — kill-auto-compaction reentry existence probe.
#
# Inputs:  none (path is fixed by the design — the orchestrator writes
#          the checkpoint to /workspace/group/.checkpoints/default.md
#          and the agent reads it from the same mount).
# Output:  single JSON line on stdout, e.g.
#            {"exists": true,  "path": "/workspace/group/.checkpoints/default.md"}
#            {"exists": false, "path": "/workspace/group/.checkpoints/default.md"}
# Exit:    0 in both cases — absence is a normal state (first-ever
#          spawn, no recent threshold-cross, operator-cleared), not a
#          failure. Non-zero exit is reserved for genuine I/O faults.
#
# Why a script: per `jbaruch/coding-policy: script-delegation`, a
# deterministic existence check is a script, not an inline code block
# in SKILL.md the agent retypes. Same input → same output, no
# reasoning involved.

set -euo pipefail

CHECKPOINT_PATH="/workspace/group/.checkpoints/default.md"

if [ -f "$CHECKPOINT_PATH" ]; then
  printf '{"exists": true, "path": "%s"}\n' "$CHECKPOINT_PATH"
else
  printf '{"exists": false, "path": "%s"}\n' "$CHECKPOINT_PATH"
fi
