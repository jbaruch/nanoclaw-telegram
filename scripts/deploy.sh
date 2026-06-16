#!/usr/bin/env bash
# Full NanoClaw deployment — single command, no manual steps.
#
# Usage: ssh nas "cd ~/nanoclaw && ./scripts/deploy.sh"
#    or: ssh nas "cd ~/nanoclaw && ./scripts/deploy.sh --tiles-only"
#    or: ssh nas "cd ~/nanoclaw && ./scripts/deploy.sh --no-cache"
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
#   8. Ensure the nanoclaw-litellm gateway is up (self-heal #609)
#
# Flags (mutually exclusive):
#   --tiles-only  Skip git pull and image rebuilds (only tile content changed).
#   --no-cache    Force `docker build --no-cache --pull` for both agent and
#                 orchestrator images. Required when an upstream npm-from-
#                 github dep in `Dockerfile.orchestrator` (e.g.
#                 `reclaim-tripit-timezones-sync`) ships a new version:
#                 BuildKit caches the `RUN npm install -g <github-repo>` layer
#                 by Dockerfile string, NOT by GitHub state, so a default
#                 deploy silently reinstalls the prior version. Also splits
#                 step 2b into separate build + `up -d --force-recreate
#                 --no-build` calls — a single `up -d --build` after a
#                 manual `--no-cache` build can resurrect a stale cached
#                 layer (observed 2026-05-02).

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
NO_CACHE=false
for arg in "$@"; do
    case "$arg" in
        --tiles-only) TILES_ONLY=true ;;
        --no-cache)   NO_CACHE=true ;;
        *)
            echo "ERROR: unknown flag '$arg' (supported: --tiles-only, --no-cache)" >&2
            exit 1
            ;;
    esac
done
if [[ "$TILES_ONLY" == true && "$NO_CACHE" == true ]]; then
    echo "ERROR: --tiles-only and --no-cache are mutually exclusive (--tiles-only skips rebuilds)." >&2
    exit 1
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
    AGENT_BUILD_FLAGS=()
    if [[ "$NO_CACHE" == true ]]; then
        AGENT_BUILD_FLAGS+=(--no-cache)
    fi
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
            ./container/build.sh "${AGENT_BUILD_FLAGS[@]}"
        else
            ./container/build.sh "$AGENT_TAG" "${AGENT_BUILD_FLAGS[@]}"
        fi
    elif [[ "$AGENT_IMAGE" == "nanoclaw-agent" ]]; then
        ./container/build.sh "${AGENT_BUILD_FLAGS[@]}"
    else
        echo "WARNING: CONTAINER_IMAGE='$AGENT_IMAGE' is not local nanoclaw-agent:*"
        echo "WARNING: ./container/build.sh will rebuild nanoclaw-agent:latest,"
        echo "WARNING: which is NOT the image the orchestrator will spawn from."
        echo "WARNING: Push/tag your own build pipeline for '$AGENT_IMAGE' separately."
        ./container/build.sh "${AGENT_BUILD_FLAGS[@]}"
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
    if [[ "$NO_CACHE" == true ]]; then
        # Split build and recreate so BuildKit can't resurrect a stale
        # cached layer. Observed 2026-05-02: a manual `docker compose
        # build --no-cache --pull nanoclaw` produced a fresh image, but
        # the subsequent `docker compose up -d --build` reused an older
        # cached layer entry and re-tagged the OLD sha as `:latest` —
        # the running container then served pre-update code. The split
        # form (`build --no-cache --pull` then `up -d --force-recreate
        # --no-build`) avoids the second build entirely.
        docker compose build --no-cache --pull nanoclaw
        docker compose up -d --force-recreate --no-build --remove-orphans nanoclaw
    else
        # `--remove-orphans` cleans up containers whose service blocks
        # were deleted from this compose. Specifically: when a sidecar
        # is moved out to its own UGOS Pro project (#610 moved
        # `nanoclaw-litellm` this way), the OLD container is still
        # alive after the compose-file change lands; without
        # `--remove-orphans`, deploy.sh leaves it running and a
        # subsequent attempt to start the UGOS project hits a port /
        # name collision. The flag has no effect on the common case
        # where no services have been removed.
        docker compose up -d --build --remove-orphans
    fi
    echo ""
else
    echo "1-2b. Skipped (--tiles-only)"
    echo ""
fi

# 3. Update tiles
#
# The previous form piped `tessl update` through `| tail -10`, which both
# truncated the error AND discarded tessl's exit code (a pipeline returns
# the LAST command's status, so `tail`'s 0 masked a failed sync). Deploys
# then reported success while applying nothing — the orchestrator kept
# running stale tiles. See `jbaruch/nanoclaw-host: post-merge-publish-watch`.
#
# Two failure modes this guards against:
#   1. tessl auto-updates its own binary in the background (see
#      ~/.tessl/auto-update.log). When that swap races this `tessl update`,
#      the sync fails with a non-zero exit. A single retry after a short
#      settle clears it (the binary is stable by the second attempt).
#   2. A genuine sync error (bad manifest, registry outage) — surface the
#      full output and HARD-FAIL the deploy rather than limp on with stale
#      tiles.
# `--accept-warnings` still installs tiles carrying advisory moderation
# verdicts (e.g. a `.env.example` flagged W008), so those do NOT fail the
# deploy — only a non-zero `tessl update` exit does.
echo "3. Updating tiles from registry..."
tessl_update_attempt=0
while :; do
    tessl_update_attempt=$((tessl_update_attempt + 1))
    # Capture status of the in-container tessl run itself, NOT a piped tail.
    if docker exec nanoclaw sh -c 'cd /app/tessl-workspace && tessl update --yes --accept-warnings 2>&1'; then
        break
    fi
    if [ "$tessl_update_attempt" -ge 2 ]; then
        echo "ERROR: 'tessl update' failed (exit non-zero) after $tessl_update_attempt attempts." >&2
        echo "       Tiles were NOT updated — aborting deploy so it can't report a false success." >&2
        echo "       Full tessl output is above. Common cause: the tessl CLI auto-updated its" >&2
        echo "       binary mid-run (see ~/.tessl/auto-update.log); re-run ./scripts/deploy.sh." >&2
        exit 1
    fi
    echo "  'tessl update' exited non-zero (attempt $tessl_update_attempt) — the tessl CLI" >&2
    echo "  auto-updates its binary in the background and can race this step; retrying in 15s..." >&2
    sleep 15
done
echo ""

# 3b. Verify every tessl.json in the repo declares mode: managed and
# floats every dependency to "latest".
#
# Per `nanoclaw-host: tessl-version-floating`, NanoClaw's lifecycle is
# dynamic-latest-loading: `tessl update` rewrites manifests in-place at
# three independent points (this script, the orchestrator's 15-min
# catch-up loop in `src/index.ts`, and the `tessl_update` MCP tool in
# `src/ipc.ts`), and `.tessl/tiles/<workspace>/<tile>/` is gitignored.
# Both invariants together rule out vendoring (mode: vendored is a lie
# when the content isn't committed) AND pinning (a literal pin produces
# a working-tree diff after every successful update that nobody commits,
# so `git pull` rolls the deployment backward against the registry).
# Every `tessl.json` MUST therefore declare `mode: managed` AND every
# `dependencies.<tile>.version` MUST be the literal string "latest" —
# the approved exception to `coding-policy: dependency-management`.
# `tessl install <tile>` writes a literal pin by default and prior
# tessl versions wrote `mode: vendored` at init, so an operator hand-
# installing a new tile (or merging a fork) can quietly reintroduce
# drift; this check fails the deploy when that happens.
echo "3b. Verifying named carve-out manifests use 'mode: managed' + 'version: latest'..."
MANIFEST_OFFENDERS=$(python3 - <<'PY'
import json, pathlib
# Read each manifest in the explicitly named carve-out set
# (`nanoclaw-host: tessl-version-floating` lists them):
#   - tessl-workspace/tessl.json  — orchestrator workspace manifest
#     (consumed by the runtime catch-up loop / tessl_update MCP tool)
#   - tessl.json                  — project-root manifest
#     (consumed by `tessl install` to populate `.tessl/tiles/` for
#      `@.tessl/RULES.md` resolution at agent runtime)
# Add to MANIFESTS in lock-step with naming a new manifest in the
# authority-of-record rule — never via globbing, which would wildcard
# the carve-out and silently auto-include manifests that the policy
# requires named explicitly.
MANIFESTS = [
    pathlib.Path("tessl-workspace/tessl.json"),
    pathlib.Path("tessl.json"),
]
bad_lines: list[str] = []
for m in MANIFESTS:
    if not m.exists():
        bad_lines.append(f"{m}: missing (named manifest in carve-out must exist)")
        continue
    try:
        data = json.loads(m.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        bad_lines.append(f"{m}: unreadable ({type(exc).__name__}: {exc})")
        continue
    if not isinstance(data, dict):
        bad_lines.append(f"{m}: root is {type(data).__name__}, expected dict")
        continue
    mode = data.get("mode")
    if mode != "managed":
        bad_lines.append(f"{m}: mode={mode!r} (must be 'managed')")
    deps = data.get("dependencies", {})
    if not isinstance(deps, dict):
        # `dependencies` set to null, list, or string would crash
        # `.items()` below and bypass the diagnostic path under
        # `set -euo pipefail`. Report the bad container shape and
        # skip the per-dep walk for this manifest.
        bad_lines.append(f"{m}: dependencies={deps!r} (must be an object)")
        continue
    for name, dep in deps.items():
        if not isinstance(dep, dict):
            # The shorthand form `"<tile>": "latest"` (string instead of
            # object) is invalid here regardless of the value, and calling
            # .get() on a non-dict would crash the verifier — report the
            # raw value with the wrapping-shape hint.
            bad_lines.append(f"{m}: {name}={dep!r} (must be a {{\"version\": \"latest\"}} object)")
            continue
        if dep.get("version") != "latest":
            bad_lines.append(f"{m}: {name}.version={dep.get('version')!r} (must be 'latest')")
if bad_lines:
    print("\n".join(bad_lines))
PY
)
if [[ -n "$MANIFEST_OFFENDERS" ]]; then
    echo "ERROR: named carve-out manifest(s) violate 'mode: managed' + 'version: latest':" >&2
    echo "$MANIFEST_OFFENDERS" | sed 's/^/  - /' >&2
    echo "Fix: edit each manifest to {\"mode\": \"managed\", \"dependencies\": {\"<tile>\": {\"version\": \"latest\"}}} and re-run deploy." >&2
    echo "Why: nanoclaw-host: tessl-version-floating (approved exception to coding-policy: dependency-management)." >&2
    exit 1
fi
echo "  ok — named carve-out manifests declare mode: managed + version: latest"
echo ""

# 3c. Verify each declared workspace tile actually MATERIALIZED at the
# registry's latest version.
#
# `tessl update` (step 3) can report success — printing `✔ Updated N
# plugins` and exiting 0 — while silently leaving a tile's on-disk
# content at the prior version, or absent entirely. When that happens
# the orchestrator goes on to mount stale/missing tile content into
# every agent container while this script reports a clean deploy. This
# step is the deploy's ground-truth check that what landed on disk is
# what the registry says is latest.
#
# Ground truth is the materialized `tile.json` on disk
# (`tessl-workspace/.tessl/plugins/<owner>/<tile>/tile.json`) — NOT
# tessl's own `✔ Updated` summary line, and NOT `tessl outdated`
# (whose "Current" column can read as up-to-date even when the files
# on disk did not actually advance). The registry's latest version
# comes from `tessl tile info`, run inside the orchestrator container
# so it shares step 3's tessl auth and registry view.
#
# Hard-fail rather than warn: a stale/missing tile reported as a
# successful deploy is exactly the silent failure this script must
# never produce. Aborting here leaves the running orchestrator
# untouched (steps 4-7 have not run yet) and names the offending
# tile(s) so the operator can resolve the install before re-deploying.
echo "3c. Verifying materialized tile versions match the registry latest..."
TILE_VERSION_OFFENDERS=$(python3 - <<'PY'
import json, os, re, subprocess

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SEMVER = re.compile(r"(\d+\.\d+\.\d+)")
PLUGINS = os.path.join("tessl-workspace", ".tessl", "plugins")

try:
    deps = json.load(
        open(os.path.join("tessl-workspace", "tessl.json"))
    ).get("dependencies", {})
except (OSError, json.JSONDecodeError) as exc:
    print(f"tessl-workspace/tessl.json: cannot read dependencies ({exc})")
    deps = {}

for ref in sorted(deps):
    owner, _, tile = ref.partition("/")
    tile_json = os.path.join(PLUGINS, owner, tile, "tile.json")
    if os.path.isfile(tile_json):
        try:
            ondisk = json.load(open(tile_json)).get("version")
        except (OSError, json.JSONDecodeError) as exc:
            print(f"{ref}: on-disk tile.json unreadable ({exc})")
            continue
    else:
        ondisk = None
    # Registry latest via in-container tessl — same auth + registry
    # view step 3's `tessl update` used.
    proc = subprocess.run(
        ["docker", "exec", "nanoclaw", "tessl", "tile", "info", ref],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        print(f"{ref}: cannot read registry latest (tessl tile info exited {proc.returncode}): {detail}")
        continue
    latest = None
    for line in ANSI.sub("", proc.stdout).splitlines():
        if "Latest Version" in line:
            m = SEMVER.search(line)
            latest = m.group(1) if m else None
            break
    if latest is None:
        print(f"{ref}: registry latest version not found in tessl tile info output")
        continue
    if ondisk != latest:
        shown = ondisk if ondisk else "(absent)"
        print(f"{ref}: on-disk={shown} registry-latest={latest}")
PY
)
if [[ -n "$TILE_VERSION_OFFENDERS" ]]; then
    echo "ERROR: 'tessl update' reported success, but these tiles did NOT land at the registry's latest version:" >&2
    echo "$TILE_VERSION_OFFENDERS" | sed 's/^/  - /' >&2
    echo "The orchestrator would mount stale or missing tile content into agent containers while this deploy reported success." >&2
    echo "Aborting before restart. Investigate why the install was skipped for the tile(s) above, resolve it, then re-run deploy." >&2
    exit 1
fi
echo "  ok — every declared workspace tile materialized at the registry latest"
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

# 4b. #305 Plan B + #456 Wave 2 — clean up legacy schedule-task rows
#     for skills that migrated to declarative `cadence:` frontmatter
#     on the admin tile.
#
# The cadence-registry from #305 Phase 2a writes
# `source = 'cadence-registry'` rows on every container spawn from the
# installed tile content. For skills that previously fired via
# operator-bootstrapped `schedule-task`-IPC rows (`source =
# 'schedule-task'`), the legacy row continues firing in parallel with
# the registry's row after the admin tile lands the new cadence
# frontmatter — that's a double-fire bug. This step DELETEs the
# specific legacy rows whose prompt invokes one of the migrated skills.
#
# Predicate is narrow to protect against false positives:
#   1. `source = 'schedule-task'` — never touches cadence-registry
#      rows that this script's later container restart will manage.
#   2. `schedule_type` matches the legacy shape: `'interval'` for
#      heartbeat/composio-fetch (30-min cadences), `'cron'` for
#      morning-brief (`0 7 * * *`). Pinned per-skill so that an
#      owner-scheduled row of the OTHER shape that happens to invoke
#      the same skill (rare but legal) stays untouched.
#   3. `prompt LIKE '%MANDATORY FIRST ACTION: Call Skill(skill:
#      "tessl__<name>")%'` — matches the specific prompt shape the
#      `schedule-task` IPC formats for operator bootstraps, not just
#      any prompt that mentions the skill. An ad-hoc owner reminder
#      ("remind me to check heartbeat") doesn't match. A future
#      orchestrator-auto-created `heartbeat-<groupFolder>` row that
#      uses a different prompt template doesn't match either.
#
# Idempotent + safe-to-ship-before-the-admin-PR-lands: when no row
# matches, `result.changes === 0` and nothing happens; when a row
# matches but the admin tile hasn't yet shipped the new cadence, the
# skill simply doesn't fire until the next deploy after the admin
# PR lands (bounded by the cadence-registry's first rebuild after
# next agent spawn).
echo "4b. Cleaning legacy schedule-task rows for cadence-migrated skills (#305 Plan B + #456)..."
docker exec nanoclaw node -e '
const Database = require("better-sqlite3");
const db = new Database("/app/store/messages.db");
// Each entry is [skill, schedule_type] — the legacy row shape the
// `schedule-task` IPC produced before the skill migrated to
// declarative `cadence:` frontmatter. Per-skill schedule_type pinning
// protects ad-hoc owner-scheduled rows of the OTHER shape from
// false-positive deletion.
const MIGRATED = [
    ["tessl__heartbeat", "interval"],
    ["tessl__composio-fetch", "interval"],
    ["tessl__morning-brief", "cron"],
    ["tessl__memory-rotation", "cron"],
    ["tessl__nightly-backup", "cron"],
    ["tessl__nightly-external-sync", "cron"],
    ["tessl__nightly-undated-task-sweep", "cron"],
    ["tessl__check-watchlist", "cron"],
    ["tessl__state-purge", "cron"],
];
// task_run_logs has FOREIGN KEY (task_id) REFERENCES scheduled_tasks(id),
// so a bare DELETE FROM scheduled_tasks fails with FOREIGN KEY
// constraint failed on every row that has any historical run logs.
// Find the offending ids first, drop their log rows, then drop the
// scheduled_tasks rows themselves — atomic via a transaction so an
// abort mid-cleanup leaves no orphan log rows pointing at a deleted
// task. The same FK pattern applies to any operator-side row drop
// that uses cancel_task on a task with run history.
let total = 0;
const findIds = db.prepare(`
    SELECT id FROM scheduled_tasks
     WHERE source = ?
       AND schedule_type = ?
       AND prompt LIKE ?
`);
const dropLogs = db.prepare(`DELETE FROM task_run_logs WHERE task_id = ?`);
const dropTask = db.prepare(`DELETE FROM scheduled_tasks WHERE id = ?`);
for (const [skill, scheduleType] of MIGRATED) {
    const ids = findIds.all(
        "schedule-task",
        scheduleType,
        `%MANDATORY FIRST ACTION: Call Skill(skill: "${skill}")%`,
    ).map((r) => r.id);
    if (ids.length === 0) continue;
    const tx = db.transaction(() => {
        for (const id of ids) {
            dropLogs.run(id);
            dropTask.run(id);
        }
    });
    tx();
    console.log(`  removed ${ids.length} legacy row(s) for ${skill} (${scheduleType}): ${ids.join(", ")}`);
    total += ids.length;
}
console.log(`  total: ${total} legacy row(s) cleaned`);
'
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
# Marker timestamp capture must be fail-open to match the
# best-effort contract documented above and `coding-policy:
# error-handling`'s "try alternatives before failing" rule. Under
# `set -euo pipefail`, a missing python3 (or a broken inline
# script) would otherwise abort the deploy mid-flight — exactly
# what the WARNING-and-degrade path for the marker is meant to
# avoid. Clearing both vars on failure also makes the later append
# block silently skip (it gates on `-n "$DEPLOY_KILLS_LOG"`).
if ! DEPLOY_KILL_START=$(python3 -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'))"); then
    echo "WARNING: python3 timestamp capture failed — heartbeat 137 suppression will fail open" >&2
    DEPLOY_KILL_START=""
    DEPLOY_KILLS_LOG=""
fi
# Build the list of agent containers to signal. The exclusion of
# infrastructure containers (orchestrator, #609 LiteLLM gateway) lives
# in scripts/exclude-infra-containers.sh — its header documents which
# names are dropped and why the litellm match targets the gateway's full
# compose container name `nanoclaw-litellm-nanoclaw-litellm-<idx>`: a bare
# `^nanoclaw-litellm$` anchor missed that name and force-killed the
# gateway every deploy, while a loose prefix would wrongly drop a
# per-group agent whose slug starts with `litellm`. The filter exits 0 on
# no match, so `|| true` here guards a `docker ps` failure only. The
# empty-list case is handled by the `[[ -z ... ]]` check on the next line.
AGENTS=$(docker ps --format '{{.Names}}' | bash scripts/exclude-infra-containers.sh || true)
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
        currently_running=$(docker ps --format '{{.Names}}' | bash scripts/exclude-infra-containers.sh || true)
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
    currently_running=$(docker ps --format '{{.Names}}' | bash scripts/exclude-infra-containers.sh || true)
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
        # race window between the `docker ps` above and this kill would
        # write to stderr (which we suppress with `2>/dev/null`) and
        # isn't in stdout, so we never log a window for a deploy that
        # didn't actually produce a 137. Without this, the marker would
        # sometimes claim a kill happened in a 30 s window where every
        # agent had already exited cleanly — false-suppress any genuine
        # OOM 137 that lands inside that fictitious window.
        KILLED=$(printf '%s\n' "${HOLDOUTS[@]}" | xargs docker kill 2>/dev/null || true)
        if [[ -n "$KILLED" && -n "$DEPLOY_KILLS_LOG" && -n "$DEPLOY_KILL_START" ]]; then
            # printf emits a literal tab via the `\t` in the format
            # string — the heartbeat parser splits on tab, and
            # busybox/ash `echo` would reinterpret the escape if
            # anyone ports the deploy off the GNU bash on the
            # Synology NAS. Same fail-open dance as DEPLOY_KILL_START
            # so a busted python3 doesn't take the deploy down.
            if DEPLOY_KILL_END=$(python3 -c "from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z'))"); then
                if ! printf '%s\t%s\n' "$DEPLOY_KILL_START" "$DEPLOY_KILL_END" >> "$DEPLOY_KILLS_LOG"; then
                    echo "WARNING: failed to append window pair to $DEPLOY_KILLS_LOG — heartbeat 137 suppression will fail open for this deploy" >&2
                fi
            else
                echo "WARNING: python3 timestamp capture failed for window end — skipping marker write for this deploy" >&2
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

# 8. Ensure the nanoclaw-litellm gateway is running (#609 LiteLLM
# migration). It is a separate UGOS Pro compose project, so step 7's
# `docker compose restart nanoclaw` does not touch it. `restart: always`
# does NOT recover it after a manual/graceful stop — Docker suppresses
# the restart policy until the next explicit start — so any stop (a UGOS
# UI stop, a stray `docker stop`, an earlier deploy that force-killed it)
# leaves the orchestrator silently bypassing to Anthropic-direct until a
# human starts it. Re-up here so every deploy self-heals the gateway.
# Idempotent: `up -d` starts it if down, no-op if already running. Run
# against the UGOS-symlinked dir so the compose project name resolves to
# `nanoclaw-litellm` (the registered project), not the repo dir basename.
#
# Verify it STAYED up, don't just fire-and-forget `up -d`: a deploy-time
# race against the concurrent orchestrator restart + agent-runner image
# rebuild has SIGKILLed the freshly-started gateway ~1s later (exit 137,
# NOT OOM). Because that external kill suppresses `restart: always`, the
# gateway then stays down and the orchestrator silently bypasses to
# anthropic-direct (degraded: no LiteLLM cost tier-down) with only a
# stream of "Connection error" lines to show for it. Re-up + recheck a
# bounded number of times so the self-heal actually heals.
LITELLM_PROJECT_DIR=/volume1/docker/nanoclaw-litellm

# Running iff the project's (single) compose container exists AND its
# Docker state is `running`. Invoked only as an `if` condition, so
# `set -e` is suspended for the body — an empty `ps -q` (nothing up) or
# a failed inspect just yields "not running" rather than aborting.
litellm_gateway_running() {
    local cid
    cid=$(cd "$LITELLM_PROJECT_DIR" && docker compose ps -q | head -1)
    [ -n "$cid" ] || return 1
    [ "$(docker inspect -f '{{.State.Status}}' "$cid")" = "running" ]
}

if [ -d "$LITELLM_PROJECT_DIR" ]; then
    echo "8. Ensuring nanoclaw-litellm gateway is up..."
    ( cd "$LITELLM_PROJECT_DIR" && docker compose up -d )
    gw_attempt=1
    gw_max=3
    while true; do
        sleep 5
        if litellm_gateway_running; then
            echo "  ok — gateway running (verified, attempt ${gw_attempt}/${gw_max})"
            break
        fi
        if [ "$gw_attempt" -ge "$gw_max" ]; then
            echo "  WARNING: nanoclaw-litellm gateway not running after ${gw_max} start attempts." >&2
            echo "  Orchestrator will bypass to anthropic-direct (degraded: no LiteLLM cost tier-down)." >&2
            echo "  Inspect: cd $LITELLM_PROJECT_DIR && docker compose logs --tail 50" >&2
            break
        fi
        echo "  gateway not up (attempt ${gw_attempt}/${gw_max}) — re-upping..."
        ( cd "$LITELLM_PROJECT_DIR" && docker compose up -d )
        gw_attempt=$((gw_attempt + 1))
    done
    echo ""
else
    echo "8. Skipped — $LITELLM_PROJECT_DIR not present (gateway not provisioned on this host)"
    echo ""
fi

echo "=== Deploy complete ==="
echo "All groups will get fresh tiles on next message."
