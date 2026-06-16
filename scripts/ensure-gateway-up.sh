#!/usr/bin/env bash
# Ensure the nanoclaw-litellm gateway (deploy.sh step 8) is up AND stays
# up. Extracted from deploy.sh so the verify-and-re-up CONTROL FLOW is
# CI-testable with `docker` stubbed — only the genuinely-unhostable
# docker calls stay unhostable; the retry/decision logic is unit-tested
# in scripts/test_ensure_gateway_up.py (coding-policy: testing-standards
# platform-bound carve-out).
#
# Why this exists: `restart: always` does NOT recover the gateway after
# an EXTERNAL kill — Docker suppresses the restart policy until an
# explicit start. A deploy-time race against the concurrent orchestrator
# restart + agent-runner image rebuild has SIGKILLed the freshly-started
# gateway ~1s later (exit 137, NOT OOM), leaving it down with the
# orchestrator silently bypassing to anthropic-direct. A fire-and-forget
# `docker compose up -d` cannot catch that, so this verifies the
# container reached `running` and re-ups a bounded number of times.
#
# Arg:  $1  the compose project dir (UGOS-symlinked so the project name
#           resolves to `nanoclaw-litellm`).
# Env (overridden by the test for a fast, deterministic run):
#   GATEWAY_MAX_ATTEMPTS  max start attempts (default 3)
#   GATEWAY_SETTLE_SECS   settle before each running-check (default 5)
# stdout: progress lines; stderr: the WARNING on exhaustion.
# exit:   0 always. A still-down gateway is NON-fatal — the orchestrator's
#         anthropic-direct bypass keeps serving, so it must not abort the
#         deploy. The stderr WARNING is the operator signal.
set -uo pipefail

PROJECT_DIR="${1:?usage: ensure-gateway-up.sh <compose-project-dir>}"
MAX_ATTEMPTS="${GATEWAY_MAX_ATTEMPTS:-3}"
SETTLE_SECS="${GATEWAY_SETTLE_SECS:-5}"

# Running iff the named `nanoclaw-litellm` service's container exists AND
# its Docker state is `running`. Querying the service by name (not the
# first id from `ps -q`) stays correct if a second service is ever added
# to the gateway's compose project.
gateway_running() {
    local cid
    cid=$(cd "$PROJECT_DIR" && docker compose ps -q nanoclaw-litellm)
    [ -n "$cid" ] || return 1
    [ "$(docker inspect -f '{{.State.Status}}' "$cid")" = "running" ]
}

attempt=1
while true; do
    ( cd "$PROJECT_DIR" && docker compose up -d )
    sleep "$SETTLE_SECS"
    if gateway_running; then
        echo "  ok — gateway running (verified, attempt ${attempt}/${MAX_ATTEMPTS})"
        exit 0
    fi
    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
        echo "  WARNING: nanoclaw-litellm gateway not running after ${MAX_ATTEMPTS} start attempts." >&2
        echo "  Orchestrator will bypass to anthropic-direct (degraded: no LiteLLM cost tier-down)." >&2
        echo "  Inspect: cd $PROJECT_DIR && docker compose logs --tail 50" >&2
        exit 0
    fi
    echo "  gateway not up (attempt ${attempt}/${MAX_ATTEMPTS}) — re-upping..."
    attempt=$((attempt + 1))
done
