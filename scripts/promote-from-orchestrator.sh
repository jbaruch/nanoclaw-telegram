#!/usr/bin/env bash
# Promote staging skills/rules to tiles — runs INSIDE the orchestrator container.
# Called by the promote_staging MCP tool via IPC.
#
# Args: $1 = group folder (e.g. "telegram_swarm")
#       $2 = tile name (e.g. "nanoclaw-admin")
#       $3 = skill name or "all" or "--rules-only"

set -euo pipefail

GROUP_FOLDER="${1:?group folder required}"
TILE_NAME="${2:?tile name required}"
MODE="${3:-all}"

REPO_DIR="/app/repo"
GROUPS_DIR="/app/groups"
TILES_DIR="$REPO_DIR/tiles"
TILE_DIR="$TILES_DIR/$TILE_NAME"
STAGING_DIR="$GROUPS_DIR/$GROUP_FOLDER/staging/$TILE_NAME"
SKILLS_SRC="$STAGING_DIR/skills"
RULES_SRC="$STAGING_DIR/rules"

if [ ! -f "$TILE_DIR/tile.json" ]; then
  echo "ERROR: $TILE_DIR/tile.json not found"
  exit 1
fi

# --- Tile placement validation ---
# Red flags that indicate a skill belongs in admin, not core or trusted.
validate_placement() {
  local skill_file="$1"
  local tile="$2"
  local canonical="$3"

  # Admin tile accepts everything
  if [ "$tile" = "nanoclaw-admin" ]; then
    return 0
  fi

  # Untrusted tile — only security rules
  if [ "$tile" = "nanoclaw-untrusted" ]; then
    if grep -qiE 'composio|gmail|calendar|tasks|schedule_task|promote|host_script|sync_tripit|fetch_trakt' "$skill_file" 2>/dev/null; then
      echo "BLOCKED: $canonical has admin-level content but target is $tile"
      echo "  Red flags found. This skill belongs in nanoclaw-admin."
      return 1
    fi
    return 0
  fi

  # Core or trusted — check for admin red flags
  if grep -qiE 'composio|gmail|googlecalendar|googletasks|sessionize|sync_tripit|fetch_trakt|promote_staging|github_backup|register_group' "$skill_file" 2>/dev/null; then
    echo "BLOCKED: $canonical has admin-level content but target is $tile"
    echo "  Found: Composio, external API, or infrastructure references."
    echo "  This skill belongs in nanoclaw-admin."
    return 1
  fi

  # Trusted — also block personal content
  if [ "$tile" = "nanoclaw-core" ]; then
    if grep -qiE '/workspace/trusted/|trusted.memory|cross.group' "$skill_file" 2>/dev/null; then
      echo "BLOCKED: $canonical references trusted workspace but target is core"
      echo "  This skill belongs in nanoclaw-trusted."
      return 1
    fi
  fi

  return 0
}

PROMOTED=0
BLOCKED=0

# --- Pull skills ---
if [ "$MODE" != "--rules-only" ]; then
  if [ "$MODE" = "all" ]; then
    SKILLS=$(ls "$SKILLS_SRC" 2>/dev/null || true)
  else
    SKILLS="$MODE tessl__$MODE"
  fi

  for skill_dir in $SKILLS; do
    [ -z "$skill_dir" ] && continue
    src="$SKILLS_SRC/$skill_dir"
    [ -d "$src" ] || continue
    [ -f "$src/SKILL.md" ] || continue

    canonical="${skill_dir#tessl__}"

    # Validate tile placement before copying
    if ! validate_placement "$src/SKILL.md" "$TILE_NAME" "$canonical"; then
      ((++BLOCKED))
      continue
    fi

    # Check for duplicates — skill must not exist in another tile
    for other_tile in "$TILES_DIR"/nanoclaw-*/; do
      other_name=$(basename "$other_tile")
      [ "$other_name" = "$TILE_NAME" ] && continue
      if [ -d "$other_tile/skills/$canonical" ]; then
        echo "BLOCKED: $canonical already exists in $other_name — skills must not duplicate across tiles"
        ((++BLOCKED))
        continue 2
      fi
    done

    dst="$TILE_DIR/skills/$canonical"
    mkdir -p "$dst"
    cp -r "$src/." "$dst/"
    echo "pulled: $canonical (from $skill_dir)"

    # Update tile.json
    python3 -c "
import json
with open('$TILE_DIR/tile.json') as f:
    tile = json.load(f)
skills = tile.setdefault('skills', {})
if '$canonical' not in skills:
    skills['$canonical'] = {'path': 'skills/$canonical/SKILL.md'}
    print('  added: $canonical')
else:
    print('  exists: $canonical')
with open('$TILE_DIR/tile.json', 'w') as f:
    json.dump(tile, f, indent=2)
    f.write('\n')
"
    ((++PROMOTED))
  done
fi

# --- Pull rules ---
if [ "$MODE" = "all" ] || [ "$MODE" = "--rules-only" ]; then
  if [ -d "$RULES_SRC" ]; then
    for rule_file in "$RULES_SRC"/*.md; do
      [ -f "$rule_file" ] || continue
      name=$(basename "$rule_file" .md)
      mkdir -p "$TILE_DIR/rules"
      cp "$rule_file" "$TILE_DIR/rules/$name.md"
      echo "pulled rule: $name"

      python3 -c "
import json
with open('$TILE_DIR/tile.json') as f:
    tile = json.load(f)
rules = tile.setdefault('rules', {})
if '$name' not in rules:
    rules['$name'] = {'rules': 'rules/$name.md'}
    print('  added: $name')
else:
    print('  exists: $name')
with open('$TILE_DIR/tile.json', 'w') as f:
    json.dump(tile, f, indent=2)
    f.write('\n')
"
      ((++PROMOTED))
    done
  fi
fi

if [ "$BLOCKED" -gt 0 ]; then
  echo ""
  echo "WARNING: $BLOCKED item(s) blocked by tile placement validation."
  echo "Review the output above. Skills with admin-level content must go to nanoclaw-admin."
fi

if [ "$PROMOTED" -eq 0 ]; then
  echo "Nothing to promote."
  exit 0
fi

# --- Lint ---
echo "Linting..."
tessl plugin lint "$TILE_DIR" || { echo "ERROR: lint failed"; exit 1; }

# --- Git commit + push ---
cd "$REPO_DIR"
TOKEN=$(grep GITHUB_TOKEN /app/.env | cut -d= -f2)
TILE_OWNER=$(grep TILE_OWNER /app/.env | cut -d= -f2)
TILE_OWNER="${TILE_OWNER:-nanoclaw}"
ASSISTANT=$(grep ASSISTANT_NAME /app/.env | cut -d= -f2)
ASSISTANT="${ASSISTANT:-Agent}"
REPO_URL=$(cd "$REPO_DIR" && git config --get remote.origin.url | sed "s|https://.*@|https://x-access-token:${TOKEN}@|; s|https://github|https://x-access-token:${TOKEN}@github|")

git config user.email "nanoclaw@bot.local"
git config user.name "$ASSISTANT"
git add "tiles/$TILE_NAME/"
if git diff --cached --quiet; then
  echo "No changes to commit."
else
  git commit -m "feat: promote $PROMOTED item(s) to $TILE_NAME from $ASSISTANT staging"
  git remote set-url origin "$REPO_URL" 2>/dev/null || \
    git remote add origin "$REPO_URL" 2>/dev/null || true
  git push origin main
fi

# --- Publish ---
echo "Publishing..."
tessl plugin publish --bump patch "$TILE_DIR" || echo "WARN: publish failed (tiles deployed via git)"

# --- Install tiles ---
echo "Installing tiles from registry..."
cd /app/tessl-workspace
tessl update \
  --yes --dangerously-ignore-security --agent claude-code 2>&1 || echo "WARN: tile update had issues"

# Kill all running agent containers so they respawn with new tiles
echo "Killing stale agent containers..."
docker ps --format '{{.ID}} {{.Names}}' | grep nanoclaw-telegram | awk '{print $1}' | xargs -r docker kill 2>/dev/null || true

echo "Done! $PROMOTED item(s) promoted, $BLOCKED blocked."
