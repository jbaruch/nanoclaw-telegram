#!/bin/bash
#
# One-time remediation for #399. Replaces the embedded-PAT origin URL
# in every group's backup-repo with the clean URL
#
#   https://github.com/jbaruch/nanoclaw.git
#
# so the token only exists at runtime via the IPC handler's
# `GIT_CONFIG_KEY_0` injection (`src/ipc.ts` case 'github_backup').
# Pre-#399 history: every group's `backup-repo/.git/config` carried
# the PAT inline as `https://x-access-token:<pat>@github.com/...`,
# visible to any shell user via `git remote -v`.
#
# Usage:  ./scripts/clean-backup-repo-origins.sh [--dry-run]
#
# Idempotent — running on a tree that's already clean is a no-op
# (every backup-repo is reported as `already_clean` and the script
# exits 0). Safe to schedule from `deploy.sh` if a future operator
# wants belt-and-suspenders enforcement, though the script ships as
# an explicit one-time step rather than a deploy-time gate.
#
# The current URL is printed with the password component redacted
# (`https://x-access-token:<redacted>@...`) so a tmux scrollback or
# terminal log of this script's output never carries a live token.
#
# Acceptance per #399:
#   - `git -C groups/<group>/backup-repo remote -v` for every group
#     returns the clean URL with no token component.
#   - `grep -rn "github_pat_" /home/jbaruch/nanoclaw` returns no hits
#     under `groups/*/backup-repo/.git/config` (only `.env` per the
#     no-secrets convention).
set -euo pipefail

CLEAN_URL="https://github.com/jbaruch/nanoclaw.git"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
GROUPS_DIR="$PROJECT_ROOT/groups"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

if [ ! -d "$GROUPS_DIR" ]; then
  echo "clean-backup-repo-origins: groups directory not found at $GROUPS_DIR — run from a NanoClaw checkout root" >&2
  exit 1
fi

redact_url() {
  # Replace the password component of a userinfo URL with `<redacted>`
  # so the printed line carries no live token. Matches `://user:pw@`.
  echo "$1" | sed -E 's,(://[^:/]+:)[^@]+(@),\1<redacted>\2,'
}

count_changed=0
count_already_clean=0
count_no_remote=0
count_no_repo=0

shopt -s nullglob
for backup_repo in "$GROUPS_DIR"/*/backup-repo; do
  if [ ! -d "$backup_repo/.git" ]; then
    count_no_repo=$((count_no_repo + 1))
    continue
  fi

  if ! current=$(git -C "$backup_repo" remote get-url origin 2>/dev/null); then
    echo "no origin remote: $backup_repo"
    count_no_remote=$((count_no_remote + 1))
    continue
  fi

  if [ "$current" = "$CLEAN_URL" ]; then
    count_already_clean=$((count_already_clean + 1))
    continue
  fi

  redacted=$(redact_url "$current")
  if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY-RUN: $backup_repo: $redacted -> $CLEAN_URL"
  else
    git -C "$backup_repo" remote set-url origin "$CLEAN_URL"
    echo "set: $backup_repo: $redacted -> $CLEAN_URL"
  fi
  count_changed=$((count_changed + 1))
done
shopt -u nullglob

echo "Summary: changed=$count_changed already_clean=$count_already_clean no_remote=$count_no_remote no_repo=$count_no_repo"
