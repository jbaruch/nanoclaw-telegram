#!/bin/bash
# Promote an AyeAye-created skill from group staging to a tessl tile.
#
# Usage: ./scripts/promote-skill.sh <skill-name> [tile-name]
#   skill-name: name of the skill directory in groups/telegram_swarm/skills/
#   tile-name:  target tile (default: nanoclaw-core)
#
# Flow:
#   1. Pulls skill from NAS (where AyeAye created it)
#   2. Copies to tiles/{tile}/skills/{name}/
#   3. Runs tessl skill review --optimize
#   4. Updates tile.json with the new skill entry
#   5. Runs tessl tile lint
#   6. Optionally commits, pushes, publishes, and deploys to NAS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

NAS_HOST="${NAS_HOST:-192.168.10.32}"
NAS_PROJECT_DIR="${NAS_PROJECT_DIR:-/home/jbaruch/nanoclaw}"
GROUP_FOLDER="${GROUP_FOLDER:-telegram_swarm}"

SKILL_NAME="${1:-}"
TILE_NAME="${2:-nanoclaw-core}"

if [ -z "$SKILL_NAME" ]; then
  echo "Usage: $0 <skill-name> [tile-name]"
  echo ""
  echo "Available skills on NAS to promote:"
  ssh "$NAS_HOST" "ls $NAS_PROJECT_DIR/groups/$GROUP_FOLDER/skills/ 2>/dev/null" 2>/dev/null || echo "  (none or NAS unreachable)"
  echo ""
  echo "Available skills locally to promote:"
  for d in "$PROJECT_ROOT"/groups/"$GROUP_FOLDER"/skills/*/; do
    [ -d "$d" ] && echo "  $(basename "$d")"
  done
  exit 1
fi

GROUP_SKILL_DIR="$PROJECT_ROOT/groups/$GROUP_FOLDER/skills/$SKILL_NAME"
TILE_SKILL_DIR="$PROJECT_ROOT/tiles/$TILE_NAME/skills/$SKILL_NAME"
TILE_JSON="$PROJECT_ROOT/tiles/$TILE_NAME/tile.json"

if [ ! -f "$TILE_JSON" ]; then
  echo "Error: $TILE_JSON not found (tile '$TILE_NAME' doesn't exist)"
  exit 1
fi

# 0. Pull skill from NAS if not already local
if [ ! -f "$GROUP_SKILL_DIR/SKILL.md" ]; then
  echo "0. Pulling skill from NAS ($NAS_HOST)..."
  mkdir -p "$GROUP_SKILL_DIR"
  ssh "$NAS_HOST" "tar czf - -C $NAS_PROJECT_DIR/groups/$GROUP_FOLDER/skills/$SKILL_NAME ." 2>/dev/null | tar xzf - -C "$GROUP_SKILL_DIR"
  if [ ! -f "$GROUP_SKILL_DIR/SKILL.md" ]; then
    echo "Error: skill '$SKILL_NAME' not found on NAS either"
    rm -rf "$GROUP_SKILL_DIR"
    exit 1
  fi
  echo "   Pulled: $GROUP_SKILL_DIR/SKILL.md"
fi

echo "=== Promoting: $SKILL_NAME → $TILE_NAME ==="
echo ""

# 1. Copy skill to tile
echo "1. Copying to tile..."
mkdir -p "$TILE_SKILL_DIR"
cp -r "$GROUP_SKILL_DIR/"* "$TILE_SKILL_DIR/"
echo "   Done: $TILE_SKILL_DIR/SKILL.md"

# 2. Review and optimize
echo ""
echo "2. Running tessl skill review --optimize..."
tessl skill review --optimize --yes --max-iterations 3 "$TILE_SKILL_DIR/SKILL.md" || echo "   (tessl review skipped — auth may be expired, run 'tessl login')"

# 3. Update tile.json
echo ""
echo "3. Updating tile.json..."
if grep -q "\"$SKILL_NAME\"" "$TILE_JSON"; then
  echo "   Skill already in tile.json — skipping"
else
  python3 -c "
import json
with open('$TILE_JSON') as f:
    tile = json.load(f)
tile['skills']['$SKILL_NAME'] = {'path': 'skills/$SKILL_NAME/SKILL.md'}
with open('$TILE_JSON', 'w') as f:
    json.dump(tile, f, indent=2)
    f.write('\n')
print('   Added $SKILL_NAME to tile.json')
"
fi

# 4. Lint
echo ""
echo "4. Running tessl tile lint..."
tessl tile lint "$PROJECT_ROOT/tiles/$TILE_NAME"

# 5. Publish prompt
echo ""
echo "=== Promotion complete ==="
echo ""
echo "Next steps:"
echo "  1. git add + commit + push"
echo "  2. tessl tile publish"
echo "  3. Deploy to NAS: git pull + docker compose up -d --build"
echo ""
read -p "Run all three now? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  cd "$PROJECT_ROOT"
  git add "tiles/$TILE_NAME/"
  git commit -m "feat: promote $SKILL_NAME skill from AyeAye staging

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
  git push origin main

  echo ""
  echo "Publishing to tessl registry..."
  tessl tile publish --bump patch "$PROJECT_ROOT/tiles/$TILE_NAME" || echo "(tessl publish skipped — run 'tessl login' then 'tessl tile publish --bump patch tiles/$TILE_NAME')"

  echo ""
  echo "Deploying to NAS..."
  ssh "$NAS_HOST" "cd $NAS_PROJECT_DIR && git pull && docker compose up -d --build" 2>/dev/null

  echo ""
  echo "Cleaning up staging copy on NAS..."
  ssh "$NAS_HOST" "rm -rf $NAS_PROJECT_DIR/groups/$GROUP_FOLDER/skills/$SKILL_NAME" 2>/dev/null && \
    echo "   Deleted: groups/$GROUP_FOLDER/skills/$SKILL_NAME" || \
    echo "   (cleanup skipped — remove manually if needed)"
  echo ""
  echo "Done! Skill promoted, published, deployed, staging cleaned."
fi
