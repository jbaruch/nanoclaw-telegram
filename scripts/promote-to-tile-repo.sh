#!/usr/bin/env bash
# Promote staged skills/rules directly to a tile's GitHub repo.
# GHA handles skill review (85%), lint, and tessl publish.
#
# Usage:
#   promote-to-tile-repo.sh <staging-dir> <tile-name> [skill-name|all|--rules-only]
#
# Environment: GITHUB_TOKEN, TILE_OWNER (defaults to "jbaruch")
#
# Runs in both contexts:
#   - Inside orchestrator container (called by IPC handler)
#   - On host Mac (called by promote-from-host.sh wrapper)

set -euo pipefail

# Load nvm if available (NAS has tessl via nvm-managed npm)
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [ -s "$NVM_DIR/nvm.sh" ]; then
  . "$NVM_DIR/nvm.sh"
fi

# gh is required for the PR-based promote flow. Fail fast with a
# clear install pointer rather than hitting a cryptic "gh: command
# not found" deep inside the script after we've already cloned the
# tile repo.
if ! command -v gh >/dev/null 2>&1; then
  echo "ERROR: promote flow requires the GitHub CLI (gh)." >&2
  echo "Install: https://cli.github.com/manual/installation" >&2
  echo "In the orchestrator container image, add gh to Dockerfile.orchestrator." >&2
  exit 1
fi

STAGING_DIR="${1:?staging directory required}"
TILE_NAME="${2:?tile name required}"
MODE="${3:-all}"

TILE_OWNER="${TILE_OWNER:-jbaruch}"
TOKEN="${GITHUB_TOKEN:?GITHUB_TOKEN required}"
ASSISTANT_NAME="${ASSISTANT_NAME:-Agent}"

SKILLS_SRC="$STAGING_DIR/skills"
RULES_SRC="$STAGING_DIR/rules"

# Cross-tile duplicate check: look at registry-installed tiles
TESSL_TILES_DIR="${TESSL_TILES_DIR:-}"

# Shared frontmatter + placement helpers. Kept in a separate file so the
# fixup-pushing script (push-staged-to-branch.sh) applies the same
# placement rules as a fresh promote — if they diverged, Copilot review
# comments could get re-pushed into the wrong tile.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./tile-repo-lib.sh
source "$SCRIPT_DIR/tile-repo-lib.sh"

# --- Clone tile repo ---
TILE_REPO_URL="https://x-access-token:${TOKEN}@github.com/${TILE_OWNER}/${TILE_NAME}.git"
TILE_REPO_DIR="/tmp/promote-${TILE_NAME}-$$"

# Clean up the temp clone on any exit path — success, failure, or
# partial-success mid-flow (e.g. PR created but Copilot summon
# failed). Without this, orphan /tmp/promote-*/ dirs pile up on the
# orchestrator. `-rf` is intentional: the dir is ours, single-purpose.
cleanup_temp() {
  rm -rf "$TILE_REPO_DIR"
}
trap cleanup_temp EXIT

echo "Cloning ${TILE_OWNER}/${TILE_NAME}..."
rm -rf "$TILE_REPO_DIR"
git clone --depth 1 "$TILE_REPO_URL" "$TILE_REPO_DIR"

PROMOTED=0
BLOCKED=0
PROMOTED_SKILLS=""

# --- Pull skills into clone ---
if [ "$MODE" != "--rules-only" ]; then
  if [ "$MODE" = "all" ]; then
    if [ -d "$SKILLS_SRC" ]; then
      SKILLS=$(ls "$SKILLS_SRC")
    else
      SKILLS=""
    fi
  else
    SKILLS="$MODE"
  fi

  for skill_dir in $SKILLS; do
    [ -z "$skill_dir" ] && continue
    src="$SKILLS_SRC/$skill_dir"
    [ -d "$src" ] || continue
    [ -f "$src/SKILL.md" ] || continue

    canonical="${skill_dir#tessl__}"

    if ! validate_placement "$src/SKILL.md" "$TILE_NAME" "$canonical"; then
      BLOCKED=$((BLOCKED + 1))
      continue
    fi

    # Cross-tile duplicate check
    if [ -n "$TESSL_TILES_DIR" ]; then
      for other_tile_dir in "$TESSL_TILES_DIR"/nanoclaw-*/; do
        other_name=$(basename "$other_tile_dir")
        [ "$other_name" = "$TILE_NAME" ] && continue
        if [ -d "$other_tile_dir/skills/$canonical" ]; then
          echo "BLOCKED: $canonical already exists in $other_name"
          BLOCKED=$((BLOCKED + 1))
          continue 2
        fi
      done
    fi

    dst="$TILE_REPO_DIR/skills/$canonical"
    mkdir -p "$dst"
    cp -r "$src/." "$dst/"
    echo "pulled: $canonical"
    PROMOTED_SKILLS="$PROMOTED_SKILLS $canonical"

    # Update tile.json (add entry if new). Values pass through argv to
    # Python — never interpolate skill names into the script body (a
    # directory with a quote would break the snippet or enable injection).
    python3 - "$TILE_REPO_DIR/tile.json" "$canonical" <<'PY'
import json, sys
tile_path, canonical = sys.argv[1], sys.argv[2]
with open(tile_path) as f:
    tile = json.load(f)
skills = tile.setdefault('skills', {})
if canonical not in skills:
    skills[canonical] = {'path': f'skills/{canonical}/SKILL.md'}
    print(f'  added: {canonical}')
else:
    print(f'  exists: {canonical}')
with open(tile_path, 'w') as f:
    json.dump(tile, f, indent=2)
    f.write('\n')
PY
    PROMOTED=$((PROMOTED + 1))
  done
fi

# --- Pull rules into clone ---
if [ "$MODE" = "all" ] || [ "$MODE" = "--rules-only" ]; then
  if [ -d "$RULES_SRC" ]; then
    for rule_file in "$RULES_SRC"/*.md; do
      [ -f "$rule_file" ] || continue
      name=$(basename "$rule_file" .md)
      mkdir -p "$TILE_REPO_DIR/rules"
      cp "$rule_file" "$TILE_REPO_DIR/rules/$name.md"
      echo "pulled rule: $name"

      python3 - "$TILE_REPO_DIR/tile.json" "$name" <<'PY'
import json, sys
tile_path, name = sys.argv[1], sys.argv[2]
with open(tile_path) as f:
    tile = json.load(f)
rules = tile.setdefault('rules', {})
if name not in rules:
    rules[name] = {'rules': f'rules/{name}.md'}
    print(f'  added: {name}')
else:
    print(f'  exists: {name}')
with open(tile_path, 'w') as f:
    json.dump(tile, f, indent=2)
    f.write('\n')
PY
      PROMOTED=$((PROMOTED + 1))
    done
  fi
fi

if [ "$BLOCKED" -gt 0 ]; then
  echo ""
  echo "WARNING: $BLOCKED item(s) blocked by tile placement validation."
fi

if [ "$PROMOTED" -eq 0 ]; then
  echo "Nothing to promote."
  exit 0
fi

# --- Local skill review + optimize (fail-fast before PR) ---
# Runs `tessl skill review --optimize` on each promoted skill before
# we create the PR. Rationale: catch quality issues locally so the
# PR starts clean, rather than pushing obvious problems and relying
# on Copilot to reject them. tessl's auto-apply of common suggestions
# (shorter prose, clearer structure) tightens the content before a
# human/bot ever reads it. No frontmatter bypass — every skill gets
# reviewed; if the review has a bad opinion on a specific skill,
# argue with it in the PR, don't pre-opt-out.
#
# tessl may not be installed in every execution context. Fall back
# with a warning instead of blocking — Copilot on the PR is the
# next gate.
if command -v tessl >/dev/null 2>&1; then
  for skill_name in $PROMOTED_SKILLS; do
    echo "reviewing: $skill_name"
    tessl skill review --optimize --yes "$TILE_REPO_DIR/skills/$skill_name"
  done
else
  echo "WARN: tessl not found, skipping local skill review (Copilot + GHA will review)"
fi

# --- Commit, push branch, open PR, request Copilot review ---
cd "$TILE_REPO_DIR"
git config user.email "nanoclaw@bot.local"
git config user.name "$ASSISTANT_NAME"
git add -A
if git diff --cached --quiet; then
  echo "Tile repo already up to date."
  echo "Done! $PROMOTED promoted, $BLOCKED blocked."
  exit 0
fi

# Branch named with UTC timestamp + tile so concurrent promotes don't
# collide. Short enough to scan in the GitHub UI.
BRANCH="promote/$(date -u +%Y%m%dT%H%M%SZ)-${TILE_NAME}"
git checkout -b "$BRANCH"
COMMIT_MSG="feat: promote $PROMOTED item(s) from $ASSISTANT_NAME staging"
git commit -m "$COMMIT_MSG"

# Push branch. gh inherits GITHUB_TOKEN → GH_TOKEN via the env below.
git push -u origin "$BRANCH"

# Print the branch name on its own line so the agent can parse it out of
# stdout and feed it back to the `push_staged_to_branch` MCP tool for
# fixup commits.
echo "Branch: $BRANCH"

PR_BODY="Promoted by nanoclaw's promote-to-tile-repo.sh. $PROMOTED item(s) staged by $ASSISTANT_NAME.

## Review gate
Copilot review requested below. Merge after the review is clean and any findings are addressed. GHA (tessl publish, lint) runs at merge time on main.

## Iteration
Fixups land on THIS branch via the \`push_staged_to_branch\` MCP tool: read PR comments → fix in staging → call the tool with branch \`$BRANCH\`. Restage-and-re-promote is the fallback; it opens a new PR."

# --repo pinned explicitly per repo-chain.md: gh otherwise defaults to
# the upstream fork in some environments and would leak tile updates
# to the wrong repo.
PR_URL=$(GH_TOKEN="$TOKEN" gh pr create \
  --repo "$TILE_OWNER/$TILE_NAME" \
  --base main \
  --head "$BRANCH" \
  --title "$COMMIT_MSG" \
  --body "$PR_BODY")

echo "PR opened: $PR_URL"

# Summon Copilot. `summon_copilot_or_warn` lives in tile-repo-lib.sh so
# the fixup-push script can call the same code — consistency here matters
# because every branch update should get the same reviewer treatment.
PR_NUMBER="${PR_URL##*/}"
GH_TOKEN="$TOKEN" summon_copilot_or_warn "$TILE_OWNER" "$TILE_NAME" "$PR_NUMBER"

echo "Done! $PROMOTED promoted, $BLOCKED blocked."
