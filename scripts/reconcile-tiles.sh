#!/usr/bin/env bash
# Reconcile tiles: compare tile GitHub repos vs registry-installed in
# orchestrator container.
#
# Reports:
#   - Version mismatches and missing tiles
#   - Content-hash drift between GitHub HEAD and the installed copy in
#     the orchestrator container, even when version numbers match (catches
#     same-version-republished-with-different-content and legacy-path
#     stale-install)
#   - Other `.tessl/tiles` directories visible inside the container that
#     are NOT the canonical install root (operator-inspecting these
#     would see stale content)
#   - Pending staging on NAS that hasn't been promoted yet
#   - Latest GHA workflow conclusion per tile repo
#
# Closes #241 (content-hash + canonical path + richer drift) and
# #242 (silent exit when TILE_OWNER absent from .env).
#
# Usage:
#   ./scripts/reconcile-tiles.sh
#
# Env overrides:
#   TILE_OWNER          GitHub owner of the tile repos (default: jbaruch)
#   NANOCLAW_TESSL_ROOT Canonical install root inside the orchestrator
#                       container (default: /app/tessl-workspace/.tessl/tiles)

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Resolve TILE_OWNER with a reachable fallback chain. Earlier this
# script grep'd `.env` under `set -euo pipefail` and the grep failure
# (when `TILE_OWNER` isn't pinned) tripped the strict-mode exit before
# the `${VAR:-jbaruch}` fallback could fire — silent exit 0 with empty
# output, indistinguishable from "all clean" (issue #242). Now: prefer
# the env var, fall back to `.env`, fall back to the canonical owner.
TILE_OWNER_VAL="${TILE_OWNER:-$(grep '^TILE_OWNER=' "$PROJECT_ROOT/.env" 2>/dev/null | cut -d= -f2 || true)}"
TILE_OWNER_VAL="${TILE_OWNER_VAL:-jbaruch}"

# Canonical install root inside the orchestrator container. Override
# via `NANOCLAW_TESSL_ROOT` only when the install layout has been
# deliberately moved — otherwise leave the default. The script reports
# this path explicitly so the operator knows which tree was inspected.
INSTALL_ROOT="${NANOCLAW_TESSL_ROOT:-/app/tessl-workspace/.tessl/tiles}"

TILES="nanoclaw-admin nanoclaw-core nanoclaw-trusted nanoclaw-untrusted nanoclaw-host"
ISSUES=0

echo "=== Tile Reconciliation ==="
echo "Tile owner:           $TILE_OWNER_VAL"
echo "Canonical install:    nas:/container[nanoclaw]:$INSTALL_ROOT"
echo ""

# Warn on non-canonical orchestrator-level tessl-workspace installs
# inside the container. These don't fail the reconcile — they're a
# heads-up that an operator running ad-hoc grep / cat against the
# wrong path will see stale content.
#
# Match shape: `*/tessl-workspace/.tessl/tiles` anywhere under /app
# that ISN'T the canonical INSTALL_ROOT. The bare-shape filter
# excludes per-group container installs at
# `/app/data/sessions/<group>/.claude/.tessl/tiles`, which are
# legitimate runtime installs for the spawned agent containers
# (different layer; see container-runner.ts session mounts) — those
# can drift on their own and warrant their own reconcile, but they
# are not what the orchestrator's `tessl update` writes.
LEGACY_PATHS=$(nas "docker exec nanoclaw find /app -maxdepth 5 -type d -path '*/tessl-workspace/.tessl/tiles' 2>/dev/null" || true)
if [ -n "$LEGACY_PATHS" ]; then
  OTHER_PATHS=$(echo "$LEGACY_PATHS" | grep -v "^$INSTALL_ROOT$" || true)
  if [ -n "$OTHER_PATHS" ]; then
    echo "WARN: non-canonical tessl-workspace install detected in the container:"
    echo "$OTHER_PATHS" | sed 's/^/  /'
    echo "  Operators inspecting these will see stale content. This script"
    echo "  reconciles only against $INSTALL_ROOT."
    echo ""
  fi
fi

# Compute the content hash of a tile at a specific registered version.
# Uses `tessl install <name>@<version>` into a throwaway workspace so
# the hash is rooted in the registry's canonical tarball for that
# version — not GitHub HEAD, which races ahead via `patch-version-publish`'s
# auto-bump commits and would produce false-positive drift reports.
#
# We hash the union of `tile.json`, every file under `rules/`, and
# every file under `skills/` — that's the entire surface the registry
# tarball ships (per `rules/context-artifacts.md`) and the entire
# surface the installed copy exposes to the agent. Returns the SHA256
# hex digest of a sorted manifest of `<sha256>  <relpath>` lines, or
# the literal string `ERR` on failure.
registry_tile_hash() {
  local tile="$1" version="$2"
  local tmp; tmp="$(mktemp -d -t reconcile-tile-XXXXXX)"
  (
    cd "$tmp"
    cat > tessl.json <<EOF
{"name":"reconcile-verify","mode":"managed","dependencies":{}}
EOF
    if ! tessl install "$TILE_OWNER_VAL/$tile@$version" --yes --dangerously-ignore-security >/dev/null 2>&1; then
      echo "ERR"
      return
    fi
    local installed_dir=".tessl/tiles/$TILE_OWNER_VAL/$tile"
    if [ ! -d "$installed_dir" ]; then
      echo "ERR"
      return
    fi
    cd "$installed_dir"
    {
      [ -f tile.json ] && shasum -a 256 tile.json
      find rules skills -type f \( -name '*.md' -o -name '*.json' -o -name '*.py' -o -name '*.sh' \) 2>/dev/null \
        | sort \
        | xargs shasum -a 256 2>/dev/null
    } | sort | shasum -a 256 | awk '{print $1}'
  )
  rm -rf "$tmp"
}

# Same hash function applied to the container's installed copy of the
# tile. Operates inside the orchestrator container via the existing
# `nas` SSH helper. Mirrors the github_tile_hash file selection
# exactly so the two are directly comparable.
installed_tile_hash() {
  local tile="$1"
  nas "docker exec nanoclaw sh -c 'cd $INSTALL_ROOT/$TILE_OWNER_VAL/$tile 2>/dev/null && {
    [ -f tile.json ] && sha256sum tile.json
    find rules skills -type f \( -name \"*.md\" -o -name \"*.json\" -o -name \"*.py\" -o -name \"*.sh\" \) 2>/dev/null \
      | sort \
      | xargs sha256sum 2>/dev/null
  } | sort | sha256sum | awk \"{print \\\$1}\"'" 2>/dev/null
}

# mtime + size of a file inside the installed tile. Used for the
# drift report so an operator can correlate "old install" vs "wrong
# content same time" at a glance.
installed_file_summary() {
  local tile="$1"
  local relpath="$2"
  nas "docker exec nanoclaw sh -c 'stat -c \"%y  %s\" $INSTALL_ROOT/$TILE_OWNER_VAL/$tile/$relpath 2>/dev/null'" 2>/dev/null
}

echo "Tile versions (repo vs registry vs installed):"
for tile in $TILES; do
  # Get version from tile GitHub repo
  REPO_VERSION=$(gh api "repos/$TILE_OWNER_VAL/$tile/contents/tile.json" --jq '.content' 2>/dev/null | base64 -d 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['version'])" 2>/dev/null)
  if [ -z "$REPO_VERSION" ]; then
    echo "  $tile: repo NOT FOUND"
    ISSUES=$((ISSUES + 1))
    continue
  fi

  # Get version installed in orchestrator
  INSTALLED_VERSION=$(nas "docker exec nanoclaw cat $INSTALL_ROOT/${TILE_OWNER_VAL}/$tile/tile.json 2>/dev/null" 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['version'])" 2>/dev/null)
  if [ -z "$INSTALLED_VERSION" ]; then
    echo "  $tile: repo=$REPO_VERSION installed=NOT FOUND"
    ISSUES=$((ISSUES + 1))
    continue
  fi

  # Get latest version in tessl registry
  REGISTRY_VERSION=$(tessl tile info "$TILE_OWNER_VAL/$tile" 2>/dev/null | grep 'Latest Version' | awk '{print $NF}')

  if [ "$REPO_VERSION" != "$INSTALLED_VERSION" ]; then
    if [ "$REGISTRY_VERSION" = "$INSTALLED_VERSION" ]; then
      echo "  $tile: repo=$REPO_VERSION registry=$REGISTRY_VERSION installed=$INSTALLED_VERSION (repo ahead — GHA may be pending)"
    else
      echo "  $tile: repo=$REPO_VERSION registry=$REGISTRY_VERSION installed=$INSTALLED_VERSION (MISMATCH)"
      ISSUES=$((ISSUES + 1))
    fi
    continue
  fi

  # Versions match — compare content hashes between the registry's
  # canonical tarball for this version and the installed tile (#241).
  # Catches: registry republished at the same version with different
  # content, manual edits to installed files, partial unpack, install
  # layout pointing at a stale legacy directory.
  REGISTRY_HASH=$(registry_tile_hash "$tile" "$INSTALLED_VERSION")
  INSTALLED_HASH=$(installed_tile_hash "$tile" 2>/dev/null || echo "ERR")

  if [ "$REGISTRY_HASH" = "ERR" ] || [ -z "$REGISTRY_HASH" ]; then
    echo "  $tile: $REPO_VERSION (in sync; HASH-CHECK FAILED — registry install of @$INSTALLED_VERSION errored)"
    ISSUES=$((ISSUES + 1))
  elif [ "$INSTALLED_HASH" = "ERR" ] || [ -z "$INSTALLED_HASH" ]; then
    echo "  $tile: $REPO_VERSION (in sync; HASH-CHECK FAILED — installed read error)"
    ISSUES=$((ISSUES + 1))
  elif [ "$REGISTRY_HASH" = "$INSTALLED_HASH" ]; then
    echo "  $tile: $REPO_VERSION (in sync; content verified against registry@$INSTALLED_VERSION)"
  else
    # HASH-DRIFT: same version pinned, different content from what the
    # registry currently serves at that version. Surface the version
    # plus the installed tile.json's mtime so the operator can tell
    # "old install" from "wrong content same time" at a glance.
    TILEJSON_SUMMARY=$(installed_file_summary "$tile" "tile.json")
    echo "  $tile: $REPO_VERSION (HASH-DRIFT — installed content does not match registry@$INSTALLED_VERSION)"
    echo "    registry  sha256: ${REGISTRY_HASH:0:16}…"
    echo "    installed sha256: ${INSTALLED_HASH:0:16}…"
    if [ -n "$TILEJSON_SUMMARY" ]; then
      echo "    installed tile.json: $TILEJSON_SUMMARY"
    fi
    ISSUES=$((ISSUES + 1))
  fi
done

echo ""

# Check for pending staging on NAS
echo "Pending staging:"
STAGING=$(nas "find $NAS_PROJECT_DIR/groups/*/staging -type f -name '*.md' 2>/dev/null" || true)
if [ -n "$STAGING" ]; then
  echo "$STAGING" | while read -r f; do
    echo "  ${f#$NAS_PROJECT_DIR/}"
  done
else
  echo "  (empty)"
fi

echo ""

# Check GHA status for recent failures
echo "Latest GHA runs:"
for tile in $TILES; do
  result=$(gh run list --repo "$TILE_OWNER_VAL/$tile" --limit 1 --json status,conclusion --jq '.[0] | "\(.conclusion)"' 2>/dev/null)
  if [ "$result" = "failure" ]; then
    echo "  $tile: FAILED"
    ISSUES=$((ISSUES + 1))
  else
    echo "  $tile: $result"
  fi
done

echo ""
if [ "$ISSUES" -gt 0 ]; then
  echo "$ISSUES issue(s) found."
  exit 1
else
  echo "All clean."
fi
