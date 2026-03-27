#!/bin/bash
# Build the NanoClaw agent container image

set -e

# BuildKit is required for --secret mounts (tessl tile installation)
export DOCKER_BUILDKIT=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="nanoclaw-agent"
TAG="${1:-latest}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"

echo "Building NanoClaw agent container image..."
echo "Image: ${IMAGE_NAME}:${TAG}"

# Mount tessl credentials for private tile installation at build time.
# Uses --mount=type=secret so credentials never land in a Docker layer.
# Requires BuildKit (docker buildx). Falls back to no-secret build if unavailable.
TESSL_CREDS="$HOME/.tessl/api-credentials.json"
if [ -f "$TESSL_CREDS" ] && docker buildx version &>/dev/null; then
  echo "Tessl credentials available (BuildKit) — tiles will be installed"
  TESSL_SECRET_ARG="--secret id=tessl_creds,src=$TESSL_CREDS"
elif [ -f "$TESSL_CREDS" ]; then
  echo "Tessl credentials available but BuildKit not found — tiles installed without secret mount"
  TESSL_SECRET_ARG=""
else
  echo "Warning: No tessl credentials — tiles will not be installed"
  TESSL_SECRET_ARG=""
fi

${CONTAINER_RUNTIME} build ${TESSL_SECRET_ARG} --build-arg "TILE_VERSION=$(date +%s)" -t "${IMAGE_NAME}:${TAG}" .

echo ""
echo "Build complete!"
echo "Image: ${IMAGE_NAME}:${TAG}"
echo ""
echo "Test with:"
echo "  echo '{\"prompt\":\"What is 2+2?\",\"groupFolder\":\"test\",\"chatJid\":\"test@g.us\",\"isMain\":false}' | ${CONTAINER_RUNTIME} run -i ${IMAGE_NAME}:${TAG}"
