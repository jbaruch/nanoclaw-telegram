#!/bin/bash
# Build the NanoClaw agent container image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="nanoclaw-agent"
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"

# Parse args: optional positional TAG, optional --no-cache flag (any order).
# --no-cache is needed when an upstream npm-from-github dep ships a new
# version: BuildKit caches `RUN npm install -g <github-repo>` by Dockerfile
# string, NOT by GitHub state, so without invalidation a "rebuild" silently
# reinstalls the prior version. Reject any arg starting with `-` that we
# don't explicitly recognize (catches typos like `-h` that would otherwise
# silently override the tag); reject more than one positional TAG (catches
# space-typos in image references).
TAG=""
BUILD_FLAGS=()
for arg in "$@"; do
    case "$arg" in
        --no-cache)
            BUILD_FLAGS+=(--no-cache --pull)
            ;;
        -*)
            echo "ERROR: unknown flag '$arg' (supported: --no-cache)" >&2
            exit 1
            ;;
        *)
            if [[ -n "$TAG" ]]; then
                echo "ERROR: multiple positional args ('$TAG' and '$arg'); only one TAG accepted." >&2
                exit 1
            fi
            TAG="$arg"
            ;;
    esac
done
TAG="${TAG:-latest}"

echo "Building NanoClaw agent container image..."
echo "Image: ${IMAGE_NAME}:${TAG}"
if [[ ${#BUILD_FLAGS[@]} -gt 0 ]]; then
    echo "Flags: ${BUILD_FLAGS[*]}"
fi

# Tessl tiles are installed at runtime (entrypoint), not build time.
# The image only ships built-in skills and the tessl binary.
${CONTAINER_RUNTIME} build "${BUILD_FLAGS[@]}" -t "${IMAGE_NAME}:${TAG}" .

echo ""
echo "Build complete!"
echo "Image: ${IMAGE_NAME}:${TAG}"
echo ""
echo "Test with:"
echo "  echo '{\"prompt\":\"What is 2+2?\",\"groupFolder\":\"test\",\"chatJid\":\"test@g.us\",\"isMain\":false}' | ${CONTAINER_RUNTIME} run -i ${IMAGE_NAME}:${TAG}"
