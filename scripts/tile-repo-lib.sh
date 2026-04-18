#!/usr/bin/env bash
# Shared helpers for the tile-repo flow:
#   - scripts/promote-to-tile-repo.sh (opens a new PR)
#   - scripts/push-staged-to-branch.sh (appends fixups to an existing PR branch)
#
# Source-only — not meant to be run directly. Callers must have `set -e`
# active so the awk/grep invariants actually fail the script.

# Read a single frontmatter field from a SKILL.md. Prints the normalised
# value on stdout (empty if unset or no frontmatter block). Frontmatter
# is the block between the first two `---` markers at the top of the file.
#
# Robust against common YAML-ish variants:
# - `field: true`        (space after colon)
# - `field:true`         (no space after colon)
# - `field: true  `      (trailing whitespace)
# - `field: true # note` (inline comment stripped)
# - `field: "true"` / `field: 'true'` (surrounding quotes stripped)
#
# Silent-fail would be dangerous here — an author who writes
# `placement-admin-content-ok:true` thinking the bypass flag is set, then
# watches their skill get blocked anyway, has no way to diagnose the
# mismatch. Normalise defensively.
read_frontmatter_field() {
  local file="$1"
  local field="$2"
  # Match is buffered and only emitted if we saw both opening AND closing
  # `---` markers. Without the closing check, a malformed SKILL.md that
  # opens with `---` but never closes it would let body-level `field:`
  # occurrences be parsed as frontmatter — which would let an attacker
  # smuggle `placement-admin-content-ok: true` into the body and bypass
  # validation.
  awk -v f="$field" '
    NR == 1 && $0 != "---" { exit }
    NR == 1 { in_fm = 1; next }
    in_fm && $0 == "---" { closed = 1; exit }
    in_fm && !found {
      prefix_re = "^[[:space:]]*" f "[[:space:]]*:"
      if ($0 !~ prefix_re) next
      line = $0
      sub(prefix_re, "", line)
      sub("^[[:space:]]+", "", line)   # strip leading ws after colon
      # Value begins with `#` after trimming — the entire line after the
      # colon is a comment, so the field has no value.
      if (substr(line, 1, 1) == "#") { matched = ""; found = 1; next }
      # Quoted values must be parsed BEFORE stripping `#` comments, because
      # in YAML `#` inside quotes is literal, not a comment.
      if (length(line) >= 2) {
        first = substr(line, 1, 1)
        if (first == "\"" || first == "\047") {
          # Find the rightmost matching quote (naive — does not handle
          # escaped quotes, but frontmatter boolean flags never need that).
          rest = substr(line, 2)
          for (i = length(rest); i >= 1; i--) {
            if (substr(rest, i, 1) == first) {
              matched = substr(rest, 1, i - 1)
              found = 1
              next
            }
          }
          # No closing quote — fall through to unquoted handling.
        }
      }
      # Unquoted value: strip `#` comments (must be preceded by whitespace,
      # per YAML), then strip trailing whitespace.
      sub("[[:space:]]+#.*$", "", line)
      sub("[[:space:]]+$", "", line)
      matched = line
      found = 1
    }
    END {
      if (closed && found) print matched
    }
  ' "$file"
}

# Tile placement validation. Returns 0 if placement is legal for the tile,
# 1 (with a "BLOCKED: ..." line on stdout) if the skill should be rejected.
#
# Callers must pre-verify `$skill_file` exists — we don't re-check, and a
# missing file would cause grep to error out noisily (which is what we want
# rather than a silent skip).
validate_placement() {
  local skill_file="$1"
  local tile="$2"
  local canonical="$3"

  if [ "$tile" = "nanoclaw-admin" ]; then return 0; fi

  # Explicit opt-in bypass for skills that legitimately document admin-level
  # names as reference content (e.g. scrub-list entries in `ship-code`).
  # Scope: this flag ONLY skips the admin-content regex checks — tile-
  # specific structural rules (like nanoclaw-core's trusted-workspace
  # reference block) still apply. The admin-content regex can't distinguish
  # "uses these handlers" from "warns you to scrub these handlers"; the
  # flag is the author's assertion that the mentions are intentional
  # reference material. Auditable — `grep -r 'placement-admin-content-ok: true' tiles/`
  # lists every skill that opts out.
  local skip_admin_regex=false
  if [ "$(read_frontmatter_field "$skill_file" 'placement-admin-content-ok')" = "true" ]; then
    echo "  placement check: admin-content regex bypassed by frontmatter flag for $canonical"
    skip_admin_regex=true
  fi

  # Admin-content regex. We invert `grep -q`'s exit via `if`, so a successful
  # match (exit 0) enters the BLOCKED branch. grep exit 1 (no match) skips
  # the block; grep exit 2 (read error) propagates under `set -e`, which is
  # the intended loud failure.
  if [ "$tile" = "nanoclaw-untrusted" ]; then
    if ! $skip_admin_regex && grep -qiE 'composio|gmail|calendar|tasks|schedule_task|promote|host_script|sync_tripit|fetch_trakt' "$skill_file"; then
      echo "BLOCKED: $canonical has admin-level content but target is $tile"
      return 1
    fi
    return 0
  fi

  if ! $skip_admin_regex && grep -qiE 'composio|gmail|googlecalendar|googletasks|sessionize|sync_tripit|fetch_trakt|promote_staging|github_backup|register_group' "$skill_file"; then
    echo "BLOCKED: $canonical has admin-level content but target is $tile"
    return 1
  fi

  # nanoclaw-core's trusted-workspace check runs regardless of the
  # admin-content bypass — these are orthogonal concerns.
  if [ "$tile" = "nanoclaw-core" ]; then
    if grep -qiE '/workspace/trusted/|trusted.memory|cross.group' "$skill_file"; then
      echo "BLOCKED: $canonical references trusted workspace but target is core"
      return 1
    fi
  fi

  return 0
}
