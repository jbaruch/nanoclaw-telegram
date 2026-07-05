#!/usr/bin/env bash
# Filter docker container names down to NanoClaw AGENT containers.
#
# Reads container names on stdin (one per line, e.g. from
# `docker ps --format '{{.Names}}'`) and writes the names of per-group
# agent containers to stdout, dropping the orchestrator container the
# deploy agent-close path (scripts/deploy.sh) must never signal or
# force-kill.
#
# Kept:   nanoclaw-<group-slug>   per-group agent containers.
# Dropped:
#   nanoclaw                      orchestrator (no trailing dash; it is
#                                 restarted in deploy step 7, never via
#                                 the agent-close path)
#
# stdin:  container names, one per line
# stdout: agent container names, one per line (empty if none)
# exit:   0 on success (an empty agent list is success); non-zero only on
#         an awk runtime error.
#
# Single awk pass: keep lines starting with `nanoclaw-`. The orchestrator
# is named `nanoclaw` (no trailing dash) so it does not match. awk exits 0
# even when nothing matches, so callers need no `|| true` no-match guard.
set -uo pipefail

awk '/^nanoclaw-/'
