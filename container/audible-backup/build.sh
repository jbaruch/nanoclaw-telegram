#!/usr/bin/env bash
# Build the audible-backup sidecar image.
#
# The orchestrator's `audible_backup` IPC handler runs this image via
# `docker run audible-backup:latest`; see src/ipc.ts for the call site.
#
# Usage:
#   ./container/audible-backup/build.sh         # build :latest
#   ./container/audible-backup/build.sh v1.2    # build :v1.2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TAG="${1:-latest}"
IMAGE="audible-backup:${TAG}"

echo "Building ${IMAGE}..."
docker build -t "${IMAGE}" .
echo "Built ${IMAGE}"
