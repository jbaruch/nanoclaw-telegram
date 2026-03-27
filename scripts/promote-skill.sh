#!/bin/bash
# Promote an AyeAye-created skill from group staging to a tessl tile.
#
# Usage: ./scripts/promote-skill.sh <skill-name> [tile-name]
#   skill-name: name of the skill directory in groups/telegram_swarm/skills/
#   tile-name:  target tile (default: nanoclaw-core)
#
# Flow:
#   1. Copies from groups/telegram_swarm/skills/{name}/ to tiles/{tile}/skills/{name}/
#   2. Runs tessl skill review --optimize
#   3. Updates tile.json with the new skill entry
#   4. Runs tessl tile lint
#   5. Optionally publishes to registry

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SKILL_NAME="${1:-}"
TILE_NAME="${2:-nanoclaw-core}"

if [ -z "$SKILL_NAME" ]; then
  echo "Usage: $0 <skill-name> [tile-name]"
  echo ""
  echo "Available skills to promote:"
  for d in "$PROJECT_ROOT"/groups/telegram_swarm/skills/*/; do
    [ -d "$d" ] && echo "  $(basename "$d")"
  done
  exit 1
fi

GROUP_SKILL_DIR="$PROJECT_ROOT/groups/telegram_swarm/skills/$SKILL_NAME"
TILE_SKILL_DIR="$PROJECT_ROOT/tiles/$TILE_NAME/skills/$SKILL_NAME"
TILE_JSON="$PROJECT_ROOT/tiles/$TILE_NAME/tile.json"

if [ ! -f "$GROUP_SKILL_DIR/SKILL.md" ]; then
  echo "Error: $GROUP_SKILL_DIR/SKILL.md not found"
  exit 1
fi

if [ ! -f "$TILE_JSON" ]; then
  echo "Error: $TILE_JSON not found (tile '$TILE_NAME' doesn't exist)"
  exit 1
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
tessl skill review --optimize --yes --max-iterations 3 "$TILE_SKILL_DIR/SKILL.md" || true

# 3. Update tile.json
echo ""
echo "3. Updating tile.json..."
if grep -q "\"$SKILL_NAME\"" "$TILE_JSON"; then
  echo "   Skill already in tile.json — skipping"
else
  # Insert new skill entry before the closing }} of the skills object
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
echo "  git add tiles/$TILE_NAME/"
echo "  git commit -m 'feat: promote $SKILL_NAME skill from AyeAye staging'"
echo "  git push origin main"
echo "  tessl tile publish --bump patch $PROJECT_ROOT/tiles/$TILE_NAME"
echo ""
read -p "Run these now? [y/N] " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
  cd "$PROJECT_ROOT"
  git add "tiles/$TILE_NAME/"
  git commit -m "feat: promote $SKILL_NAME skill from AyeAye staging

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
  git push origin main
  tessl tile publish --bump patch "$PROJECT_ROOT/tiles/$TILE_NAME"
  echo ""
  echo "Done! Rebuild agent image to pick up the new tile version."
fi
