#!/bin/bash
# Build the NanoClaw agent container image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="nanoclaw-agent"
TAG="${1:-latest}"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"

echo "Building NanoClaw agent container image..."
echo "Image: ${IMAGE_NAME}:${TAG}"

# Get tessl API key for installing tiles at build time.
# Uses --mount=type=secret so the key never lands in a Docker layer.
TESSL_KEY="${TESSL_API_KEY:-$(tessl auth token 2>/dev/null || echo '')}"
if [ -n "$TESSL_KEY" ]; then
  echo "Tessl API key available — tiles will be installed"
  TESSL_SECRET_ARG="--secret id=tessl_key,env=TESSL_API_KEY"
  export TESSL_API_KEY="$TESSL_KEY"
else
  echo "Warning: No tessl API key — tiles will not be installed"
  TESSL_SECRET_ARG=""
fi

${CONTAINER_RUNTIME} build ${TESSL_SECRET_ARG} -t "${IMAGE_NAME}:${TAG}" .

echo ""
echo "Build complete!"
echo "Image: ${IMAGE_NAME}:${TAG}"
echo ""
echo "Test with:"
echo "  echo '{\"prompt\":\"What is 2+2?\",\"groupFolder\":\"test\",\"chatJid\":\"test@g.us\",\"isMain\":false}' | ${CONTAINER_RUNTIME} run -i ${IMAGE_NAME}:${TAG}"
