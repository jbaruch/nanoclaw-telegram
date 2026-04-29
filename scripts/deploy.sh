#!/usr/bin/env bash
# Full NanoClaw deployment — single command, no manual steps.
#
# Usage: ssh nas "cd ~/nanoclaw && ./scripts/deploy.sh"
#    or: ssh nas "cd ~/nanoclaw && ./scripts/deploy.sh --tiles-only"
#
# Steps:
#   1. Pull latest code from origin
#   2a. Rebuild agent-runner image (must precede 2b — see #69)
#   2b. Rebuild orchestrator image (`docker compose up -d --build`
#       recreates the running container as a side effect of the rebuild)
#   3. Update tiles from registry
#   4. Clear runtime skill overrides from all groups
#   5. Kill ALL running agent containers (forces fresh tile load)
#   6. Clear ALL sessions from DB
#   7. Restart orchestrator
#
# The --tiles-only flag skips the git pull and image rebuilds (for when
# only tile content changed, not source code).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Read CONTAINER_IMAGE from .env if it isn't already set in the shell,
# so this script and `docker compose` see the same source of truth.
# `docker compose` reads .env directly when interpolating
# `${CONTAINER_IMAGE:-...}`; without this lookup, a `.env`-only setting
# would be visible to compose but invisible to this script — silent
# divergence, "we rebuild what the orchestrator spawns" stops being
# true.
#
# Two design choices worth preserving:
#   1. Parse `.env` as data, NOT `source .env`. Sourcing executes the
#      file as shell code — any unexpected/malicious content runs on
#      the host. We only want one specific KEY=VALUE, not arbitrary
#      shell.
#   2. Shell-exported values WIN over `.env`. That matches `docker
#      compose`'s precedence (shell env overrides .env). Without this
#      check, `set -a + source .env` would overwrite an explicit shell
#      export — operator surprise.
if [ -f .env ] && [ -z "${CONTAINER_IMAGE:-}" ]; then
    # Match the first line that looks like `CONTAINER_IMAGE=<value>`
    # (ignoring `# CONTAINER_IMAGE=` comments and indented variants).
    # `grep -m1` stops after the first match — a single tool, no `head`
    # pipe (which would close stdin and hand grep a SIGPIPE under
    # `set -o pipefail`). `|| true` swallows the no-match exit-1 so a
    # `.env` that doesn't define CONTAINER_IMAGE leaves us at the
    # default rather than aborting the script under `set -e`.
    # Strip surrounding double or single quotes if present, the way
    # compose's .env parser does.
    raw=$(grep -m1 -E '^[[:space:]]*CONTAINER_IMAGE=' .env | sed -E 's/^[[:space:]]*CONTAINER_IMAGE=//' || true)
    if [ -n "$raw" ]; then
        # Strip matching quote pair if present
        case "$raw" in
            \"*\") raw="${raw#\"}"; raw="${raw%\"}";;
            \'*\') raw="${raw#\'}"; raw="${raw%\'}";;
        esac
        export CONTAINER_IMAGE="$raw"
    fi
fi

TILES_ONLY=false
if [[ "${1:-}" == "--tiles-only" ]]; then
    TILES_ONLY=true
fi

echo "=== NanoClaw Deploy ==="
echo ""

# 0. Guard against credentials embedded in the git remote URL (#106).
# Any `https://<user>:<token>@github.com/...` form leaks the token to
# anyone who runs `git remote -v`, reads `.git/config`, or sees a
# script's stdout when this script echoes git output. PATs with `repo`
# scope grant full read/write — leaking one is a high-severity rotate-
# now incident. Refuse to deploy until the operator switches to SSH or
# a credential helper. Pattern matches both `https://user:token@host/`
# and the GitHub-specific `x-access-token:token@host/` shape seen on
# the NAS in #106.
echo "0. Checking git remote for embedded credentials..."
if git remote -v 2>/dev/null | grep -qE 'https?://[^@/[:space:]]+:[^@/[:space:]]+@'; then
    echo "ERROR: git remote URL embeds credentials." >&2
    echo "  PATs in remote URLs leak via 'git remote -v', .git/config, and any" >&2
    echo "  script that echoes git output. Rotate the credential and switch to" >&2
    echo "  SSH:" >&2
    echo "    git remote set-url origin git@github.com:<owner>/<repo>.git" >&2
    echo "  or to a credential helper backed by a secret store. Refusing to" >&2
    echo "  deploy. See https://github.com/jbaruch/nanoclaw/issues/106" >&2
    exit 1
fi
echo ""

# 1. Pull
if [[ "$TILES_ONLY" == false ]]; then
    echo "1. Pulling latest code..."
    # `git stash` exits non-zero when there's nothing to stash — expected case on a clean tree.
    git stash 2>/dev/null || true
    git pull --no-rebase origin main
    echo ""

    # 2. Rebuild agent-runner + orchestrator.
    # Order matters: build the AGENT image first, then rebuild+restart
    # the ORCHESTRATOR. The reverse order leaves a window where the new
    # orchestrator is live but `nanoclaw-agent:latest` still points at
    # the pre-deploy image — any inbound message in that window spawns
    # an agent from the stale image (issue #69, the same stale-image
    # class of bug as #66 was meant to close).
    #
    # Doing agent first means: while build.sh runs, the OLD orchestrator
    # is still serving requests against the OLD agent image — i.e. the
    # pre-deploy steady state, not a regression. By the time the
    # orchestrator is recreated by `docker compose up -d --build`, the
    # agent image is already new.
    #
    # Orchestrator image bakes the host-side TypeScript compiled output;
    # agent-runner image bakes container-side source (MCP tools, IPC
    # bridge). Both need rebuilding after a source-code pull — previous
    # versions of this script only built the orchestrator, which left
    # the agent image stale (last observed when `nuke_session` got a
    # new `session` parameter on the schema: the schema was in git but
    # AyeAye's container still saw the old parameterless tool until
    # someone remembered to run `./container/build.sh` separately).
    #
    # Agent-image reference comes from $CONTAINER_IMAGE (the same env
    # var the orchestrator reads in src/config.ts to decide which image
    # to spawn agent containers from). When unset we default to
    # `nanoclaw-agent:latest`; otherwise we honor whatever tag the
    # operator passed (versioned, custom name, etc.). Digest-pinned
    # references like `nanoclaw-agent:latest@sha256:...` are NOT
    # supported by `./container/build.sh` and are detected below — we
    # warn and let the orchestrator continue spawning from the
    # operator-pinned image without trying to rebuild it locally.
    echo "2a. Rebuilding agent-runner..."
    AGENT_IMAGE="${CONTAINER_IMAGE:-nanoclaw-agent:latest}"
    # build.sh reads the tag from the first POSITIONAL arg, not env var
    # (`TAG="${1:-latest}"`). Passing as env var would be silently
    # ignored and default to `latest` — the exact stale-image bug this
    # PR is meant to prevent.
    if [[ "$AGENT_IMAGE" == *@sha256:* ]]; then
        # Digest-pinned reference. Docker accepts both `name:tag@sha256:...`
        # and `name@sha256:...` (digest-only, no tag) — match either via
        # `*@sha256:*`. `./container/build.sh "$tag"` doesn't accept a
        # digest and would produce an invalid `docker build -t` arg.
        # The orchestrator already pins to this exact image regardless
        # of what we rebuild locally, so warn and skip — the operator's
        # external build pipeline owns this image, not us.
        echo "WARNING: CONTAINER_IMAGE='$AGENT_IMAGE' is digest-pinned; skipping local agent rebuild."
        echo "WARNING: The orchestrator will continue spawning from the pinned digest as-is."
    elif [[ "$AGENT_IMAGE" == nanoclaw-agent:* ]]; then
        AGENT_TAG="${AGENT_IMAGE#nanoclaw-agent:}"
        # Guard against CONTAINER_IMAGE="nanoclaw-agent:" (trailing colon,
        # empty tag). build.sh's `${1:-latest}` only defaults on UNSET/
        # missing — an explicitly-passed empty string stays empty and
        # would build the invalid reference `nanoclaw-agent:`. Fall back
        # to latest with a warning so the operator notices the typo.
        if [[ -z "$AGENT_TAG" ]]; then
            echo "WARNING: CONTAINER_IMAGE='$AGENT_IMAGE' has an empty tag; building nanoclaw-agent:latest instead."
            ./container/build.sh
        else
            ./container/build.sh "$AGENT_TAG"
        fi
    elif [[ "$AGENT_IMAGE" == "nanoclaw-agent" ]]; then
        ./container/build.sh
    else
        echo "WARNING: CONTAINER_IMAGE='$AGENT_IMAGE' is not local nanoclaw-agent:*"
        echo "WARNING: ./container/build.sh will rebuild nanoclaw-agent:latest,"
        echo "WARNING: which is NOT the image the orchestrator will spawn from."
        echo "WARNING: Push/tag your own build pipeline for '$AGENT_IMAGE' separately."
        ./container/build.sh
    fi
    echo ""

    # `docker compose up -d --build` rebuilds the orchestrator image AND
    # recreates the running container as a side effect — the explicit
    # restart in step 7 is a separate clean-state pass after steps 3-6
    # have mutated DB and FS, not a duplicate of this one.
    #
    # Residual race window: while THIS step's image build is in flight,
    # the OLD orchestrator stays up and may spawn agents from the agent
    # tag (`$AGENT_IMAGE`, defaulting to nanoclaw-agent:latest but
    # operator-overridable via $CONTAINER_IMAGE) — which step 2a JUST
    # repointed at the new agent image. So during the 2b build window
    # the system runs with
    # OLD orchestrator + NEW agent, NOT pre-deploy steady state. We
    # accept this asymmetry per #69's Option 1: the pre-fix bug had the
    # orchestrator already recreated to the NEW image while the agent
    # image was still OLD — exactly the contract violation #66 was
    # meant to close. The post-fix old-orchestrator/new-agent combo is
    # the kind of asymmetry any rolling deploy temporarily exposes,
    # not a fresh-spawn-from-stale-agent. Option 3 — pre-build both
    # images then atomic-swap — would close the residual race entirely
    # but adds complexity not worth the cost for a personal deploy.
    echo "2b. Rebuilding orchestrator..."
    docker compose up -d --build
    echo ""
else
    echo "1-2b. Skipped (--tiles-only)"
    echo ""
fi

# 3. Update tiles
echo "3. Updating tiles from registry..."
docker exec nanoclaw sh -c 'cd /app/tessl-workspace && tessl update --yes --dangerously-ignore-security 2>&1' | tail -10
echo ""

# 3b. Verify no literal pins crept into tessl-workspace/tessl.json.
#
# Per `nanoclaw-host: tessl-version-floating` (registry 0.1.25+), every
# `dependencies.<tile>.version` in this manifest MUST be the literal
# string "latest" — an approved exception to `coding-policy:
# dependency-management` for this one file. `tessl install <tile>`
# writes a literal pin by default, so an operator hand-installing a
# new tile (or merging a fork) can quietly reintroduce drift; this
# check fails the deploy when that happens.
echo "3b. Verifying tessl-workspace/tessl.json uses 'latest' for every dependency..."
PINNED_OFFENDERS=$(python3 - <<'PY'
import json, sys, pathlib
manifest = pathlib.Path("tessl-workspace/tessl.json")
data = json.loads(manifest.read_text())
bad = [
    name for name, dep in data.get("dependencies", {}).items()
    if dep.get("version") != "latest"
]
if bad:
    print("\n".join(bad))
PY
)
if [[ -n "$PINNED_OFFENDERS" ]]; then
    echo "ERROR: tessl-workspace/tessl.json has non-'latest' pins:" >&2
    echo "$PINNED_OFFENDERS" | sed 's/^/  - /' >&2
    echo "Fix: edit each entry to {\"version\": \"latest\"} and re-run deploy." >&2
    echo "Why: nanoclaw-host: tessl-version-floating (approved exception to coding-policy: dependency-management)." >&2
    exit 1
fi
echo "  ok — all dependencies float to latest"
echo ""

# 4. Clear runtime skill overrides from all groups
# NOTE: staging/ is NOT cleared here — that's verify-tiles' job after promotion.
echo "4. Clearing runtime skill overrides..."
OVERRIDE_COUNT=0
for group_dir in groups/*/; do
    skills_dir="${group_dir}skills"
    if [[ -d "$skills_dir" ]] && [[ -n "$(ls -A "$skills_dir" 2>/dev/null)" ]]; then
        echo "  cleaning: $skills_dir"
        rm -rf "${skills_dir:?}"/*
        OVERRIDE_COUNT=$((OVERRIDE_COUNT + 1))
    fi
done
echo "  cleaned $OVERRIDE_COUNT group(s) with overrides"
echo ""

# 5. Gracefully close agent containers (#221).
#
# Pre-#221 this step ran `docker kill` on every nanoclaw-* container
# unconditionally — exit 137 across every in-flight conversation
# turn and scheduled-task run, even when the agent was 100ms from
# completing its reply. The rationale was "force fresh tile load
# after rebuild", but for the typical agent that's mid-query, the
# right behaviour is "let it finish its current turn, exit cleanly,
# next message spawns a fresh container with new tiles".
#
# Pattern: write the agent-runner's `_close` IPC sentinel into each
# agent's input dir, give them a grace window to exit naturally,
# then force-kill any holdouts (genuinely-stuck agents in long tool
# calls beyond the grace). User-visible: at most one stale-tile
# turn per group per deploy (the in-flight turn), instead of every
# in-flight turn destroyed mid-stream.
#
# `_close` semantics live in `container/agent-runner/src/index.ts`
# (search `IPC_INPUT_CLOSE_SENTINEL`). The agent-runner polls every
# IPC_POLL_MS (~0.5s), so signal-to-exit latency is dominated by
# the current SDK turn, not the polling loop. 30s grace is generous
# for typical turns; longer tool calls fall through to the
# pre-#221 force-kill path.
#
# Mount discovery uses `docker inspect` to find the bind mount whose
# Destination is `/workspace/ipc/input`. That destination path is a
# constant in the agent-runner (`IPC_INPUT_DIR`), so all agents
# share it regardless of group/session — much simpler than
# reverse-engineering the group folder + session name from the
# container's name suffix (which would also be ambiguous: group
# folders containing underscores get sanitised to dashes in the
# container name, losing the original spelling).
echo "5. Gracefully closing agent containers..."
# Window-pair marker for the heartbeat skill's 137-cascade
# suppression (jbaruch/nanoclaw#249). Each step-5 invocation that
# actually issues a force-kill appends one TSV line to
# data/host-logs/deploy-kills.log:
#     <start_iso>\t<end_iso>
# heartbeat-checks.py reads this from `/workspace/host-logs/` —
# which is mounted read-only **only into the main/admin
# container** per the `isMain` block in
# `src/container-runner.ts::buildVolumeMounts` (host-logs are
# inherently cross-chat and must not leak to trusted/untrusted
# tiles), and that's exactly where the admin-tile heartbeat skill
# runs. Trusted/untrusted agents see no `/workspace/host-logs`
# and never run the suppression code path.
# Use python3 for the millisecond-precision UTC timestamp because
# `date -u '+%Y-%m-%dT%H:%M:%S.%3NZ'` requires GNU `%3N`; on a
# BSD/busybox `date` (likely if anyone ports the deploy off the
# Synology NAS) the literal `%3N` would propagate and the
# heartbeat consumer's `datetime.fromisoformat` would silently
# drop the row instead of suppressing it.
# Generation is best-effort with a stderr warning rather than
# `set -e`-fatal: the marker is auxiliary alert-suppression
# plumbing, NOT a deploy precondition. If host-logs is unwritable
# we want the deploy to keep going (rejecting writes loudly
# enough that the operator notices) and the heartbeat to fail
# open back to surfacing 137s — strictly worse than suppression
# but observable, vs. a wedged deploy.
DEPLOY_KILLS_LOG="data/host-logs/deploy-kills.log"
DEPLOY_KILLS_DIR="$(dirname "$DEPLOY_KILLS_LOG")"
if ! mkdir -p "$DEPLOY_KILLS_DIR"; then
    echo "WARNING: cannot create $DEPLOY_KILLS_DIR — heartbeat 137 suppression will fail open" >&2
    DEPLOY_KILLS_LOG=""
fi
DEPLOY_KILL_START=$(python3 -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'))")
# `grep` exits 1 when no agents match — the empty-string case is
# handled by the `[[ -z ... ]]` check on the next line.
AGENTS=$(docker ps --format '{{.Names}}' | grep '^nanoclaw-' | grep -v '^nanoclaw$' || true)
if [[ -z "$AGENTS" ]]; then
    echo "  no agent containers running"
else
    AGENT_COUNT=$(echo "$AGENTS" | wc -l | tr -d ' ')
    echo "  $AGENT_COUNT agent(s) running — sending _close sentinel"

    # The grace-window poll and the final force-kill must operate
    # on the SAME set of names this loop signals — NOT on a fresh
    # `docker ps` later. A new agent that spawns mid-deploy (e.g.
    # an inbound message during the grace window) is unrelated to
    # this deploy's "give in-flight work a chance to finish" intent
    # and must NOT be force-killed at the end. Track the original
    # set in `ORIGINAL_AGENTS` and re-derive holdouts as
    # `intersect(ORIGINAL_AGENTS, currently-running)`.
    declare -A ORIGINAL_AGENTS=()
    SIGNALED_COUNT=0
    UNRESOLVED=()
    while IFS= read -r container; do
        [[ -z "$container" ]] && continue
        ORIGINAL_AGENTS["$container"]=1
        # Resolve the agent's IPC input dir on the host. The
        # template extracts the Source (host path) of the mount
        # whose Destination matches the agent-runner's constant. A
        # blank result means either the container exited between
        # the `ps` and this `inspect` (benign race) or it doesn't
        # have the expected mount (shouldn't happen for
        # nanoclaw-* but defensive).
        IPC_INPUT_HOST=$(docker inspect "$container" \
            --format '{{range .Mounts}}{{if eq .Destination "/workspace/ipc/input"}}{{.Source}}{{end}}{{end}}' \
            2>/dev/null || true)
        if [[ -z "$IPC_INPUT_HOST" || ! -d "$IPC_INPUT_HOST" ]]; then
            # Mount not resolvable — fall through to force-kill.
            UNRESOLVED+=("$container")
            continue
        fi
        # `touch` is the idiomatic empty-file create; the
        # agent-runner only checks for existence, not contents. A
        # touch failure (permissions, transient FS error) means
        # the agent will NOT see the sentinel and will run until
        # the grace window expires — track it in UNRESOLVED so
        # the operator-visible "signaled X/Y" count and the
        # eventual force-kill story are accurate. Without this,
        # a silently-failed touch would be reported as success
        # while the container kept running until force-killed.
        if touch "$IPC_INPUT_HOST/_close" 2>/dev/null; then
            SIGNALED_COUNT=$((SIGNALED_COUNT + 1))
        else
            UNRESOLVED+=("$container")
        fi
    done <<< "$AGENTS"
    echo "  signaled $SIGNALED_COUNT/$AGENT_COUNT with _close sentinel"
    if (( ${#UNRESOLVED[@]} > 0 )); then
        echo "  WARN: ${#UNRESOLVED[@]} container(s) could not be signaled (mount unresolved or _close write failed) — will force-kill after grace"
    fi

    # Poll for natural exit. 30s covers typical turn completion
    # (SDK reply + cleanup); longer tool calls (large file ops,
    # slow MCP calls) hit the force-kill below — same destructive
    # behaviour as pre-#221, just narrowed to the genuinely-stuck
    # minority instead of every running agent.
    #
    # Compare against ORIGINAL_AGENTS, not `docker ps` directly —
    # a freshly-spawned agent (inbound message mid-deploy) is none
    # of this loop's business and must not extend the grace window
    # nor land in the holdout-kill set.
    GRACE_SECONDS=30
    POLL_INTERVAL=2
    elapsed=0
    while (( elapsed < GRACE_SECONDS )); do
        still_running_count=0
        currently_running=$(docker ps --format '{{.Names}}' | grep '^nanoclaw-' | grep -v '^nanoclaw$' || true)
        while IFS= read -r name; do
            [[ -z "$name" ]] && continue
            if [[ -n "${ORIGINAL_AGENTS[$name]:-}" ]]; then
                still_running_count=$((still_running_count + 1))
            fi
        done <<< "$currently_running"
        if (( still_running_count == 0 )); then
            echo "  all signaled agents exited gracefully after ${elapsed}s"
            break
        fi
        sleep "$POLL_INTERVAL"
        elapsed=$((elapsed + POLL_INTERVAL))
    done

    # Force-kill holdouts FROM THE ORIGINAL SET only. Pre-#221
    # every container was killed unconditionally; post-#221 only
    # the genuinely-stuck minority of the original set is
    # destroyed. A container may exit between this `docker ps`
    # and the kill below — `docker kill` on a dead container is a
    # benign race, not a failure.
    HOLDOUTS=()
    currently_running=$(docker ps --format '{{.Names}}' | grep '^nanoclaw-' | grep -v '^nanoclaw$' || true)
    while IFS= read -r name; do
        [[ -z "$name" ]] && continue
        if [[ -n "${ORIGINAL_AGENTS[$name]:-}" ]]; then
            HOLDOUTS+=("$name")
        fi
    done <<< "$currently_running"
    if (( ${#HOLDOUTS[@]} > 0 )); then
        echo "  ${#HOLDOUTS[@]} agent(s) from the original set didn't exit in ${GRACE_SECONDS}s — force-killing"
        # Capture stdout — `docker kill` echoes the names of containers
        # it actually killed. A holdout that exited gracefully in the
        # race window between the `docker ps` above and this kill prints
        # to stderr and isn't in stdout, so we never log a window for a
        # deploy that didn't actually produce a 137. Without this, the
        # marker would sometimes claim a kill happened in a 30 s window
        # where every agent had already exited cleanly — false-suppress
        # any genuine OOM 137 that lands inside that fictitious window.
        KILLED=$(printf '%s\n' "${HOLDOUTS[@]}" | xargs docker kill 2>/dev/null || true)
        if [[ -n "$KILLED" && -n "$DEPLOY_KILLS_LOG" ]]; then
            # printf emits a literal tab via the `\t` in the format
            # string — the heartbeat parser splits on tab, and
            # busybox/ash `echo` would reinterpret the escape if
            # anyone ports the deploy off the GNU bash on the
            # Synology NAS. Same python3 dance as DEPLOY_KILL_START
            # so the timestamps stay in lockstep.
            DEPLOY_KILL_END=$(python3 -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'))")
            if ! printf '%s\t%s\n' "$DEPLOY_KILL_START" "$DEPLOY_KILL_END" >> "$DEPLOY_KILLS_LOG"; then
                echo "WARNING: failed to append window pair to $DEPLOY_KILLS_LOG — heartbeat 137 suppression will fail open for this deploy" >&2
            fi
        fi
    fi
fi
echo ""

# 6. Clear sessions
echo "6. Clearing all sessions..."
sqlite3 store/messages.db 'DELETE FROM sessions'
CLEARED=$(sqlite3 store/messages.db 'SELECT changes()')
echo "  cleared $CLEARED sessions"
echo ""

# 7. Restart orchestrator (final clean-state restart).
# Step 2b's `up -d --build` already recreated the container on the new
# image, but steps 3-6 mutated DB state (sessions cleared, agents killed,
# tiles refreshed). Restart again so the running orchestrator process
# loads from a clean post-cleanup state instead of running with whatever
# in-memory caches were warm before steps 3-6. Cheap (no rebuild —
# `restart` reuses the image from 2b); avoids subtle staleness bugs.
echo "7. Restarting orchestrator..."
docker compose restart nanoclaw
echo ""

echo "=== Deploy complete ==="
echo "All groups will get fresh tiles on next message."
