#!/usr/bin/env bash
# Filter docker container names down to NanoClaw AGENT containers.
#
# Reads container names on stdin (one per line, e.g. from
# `docker ps --format '{{.Names}}'`) and writes the names of per-group
# agent containers to stdout, dropping infrastructure containers the
# deploy agent-close path (scripts/deploy.sh) must never signal or
# force-kill.
#
# Kept:   nanoclaw-<group-slug>   per-group agent containers, INCLUDING
#                                 a slug that starts with `litellm`
#                                 (e.g. `nanoclaw-litellm-fans`) — only
#                                 the gateway's own compose container is
#                                 dropped, not every `litellm`-ish name.
# Dropped:
#   nanoclaw                      orchestrator (no trailing dash; it is
#                                 restarted in deploy step 7, never via
#                                 the agent-close path)
#   nanoclaw-litellm-nanoclaw-litellm-<idx>
#                                 the #609 LiteLLM gateway. It runs as a
#                                 separate UGOS Pro compose project named
#                                 `nanoclaw-litellm` with a single service
#                                 `nanoclaw-litellm`, so docker names the
#                                 container `nanoclaw-litellm-nanoclaw-litellm-1`
#                                 (project + service + replica index). It
#                                 has no `/workspace/ipc/input` mount, so
#                                 the agent-close path would force-kill it
#                                 every deploy. Match the full
#                                 project+service+index shape (index left
#                                 flexible for a recreate/scale bump): a
#                                 bare `^nanoclaw-litellm$` anchor missed
#                                 the compose-suffixed name, and a loose
#                                 `^nanoclaw-litellm` prefix would wrongly
#                                 drop a per-group agent whose slug starts
#                                 with `litellm`. If a second service is
#                                 ever added to the gateway's compose
#                                 project, extend this pattern.
#
# stdin:  container names, one per line
# stdout: agent container names, one per line (empty if none)
# exit:   0 on success (an empty agent list is success); non-zero only on
#         an awk runtime error.
#
# Single awk pass: keep lines starting with `nanoclaw-` AND not matching
# the gateway's compose container name. awk exits 0 even when nothing
# matches, so callers need no `|| true` no-match guard.
set -uo pipefail

awk '/^nanoclaw-/ && !/^nanoclaw-litellm-nanoclaw-litellm-[0-9]+$/'
