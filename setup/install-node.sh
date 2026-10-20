#!/usr/bin/env bash
# Install .nvmrc's exact Node release. stdout: its bin directory; progress
# and actionable failures: stderr. Does not edit the caller's shell profile.
set -euo pipefail

node_setup_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
node_setup_version=$(<"$node_setup_root/.nvmrc")
if [[ ! "$node_setup_version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Invalid .nvmrc: set an exact Node release version, then retry setup/install-node.sh." >&2
  exit 2
fi

# No scanner tracks this script pin: check monthly, renewing tag and commit
# together from https://github.com/nvm-sh/nvm/releases. Node renews via .nvmrc.
node_setup_nvm_tag=v0.40.8
node_setup_nvm_commit=a885b885fef16fac4bc544188fb25e9e37ae83e8
node_setup_cache="${NANOCLAW_NODE_BOOTSTRAP_DIR:-$HOME/.local/share/nanoclaw-node-bootstrap}"
export NVM_DIR="$node_setup_cache/nvm-$node_setup_nvm_tag"

if mkdir -p "$node_setup_cache"; then
  :
else
  node_setup_rc=$?
  echo "Cannot create the managed Node bootstrap directory (exit $node_setup_rc). Check its parent permissions and retry." >&2
  exit "$node_setup_rc"
fi
if [ ! -f "$NVM_DIR/nvm.sh" ]; then
  if git clone --depth 1 --branch "$node_setup_nvm_tag" https://github.com/nvm-sh/nvm.git "$NVM_DIR" >&2; then
    :
  else
    node_setup_rc=$?
    echo "Cannot download the pinned Node installer (git exit $node_setup_rc). Check Git, network access and the bootstrap directory, then retry." >&2
    exit "$node_setup_rc"
  fi
fi
node_setup_rc=0
node_setup_actual=$(git -C "$NVM_DIR" rev-parse HEAD) || node_setup_rc=$?
if [ "$node_setup_rc" -ne 0 ]; then
  echo "Cannot verify the managed Node installer (git exit $node_setup_rc). Move the incomplete bootstrap directory aside and retry." >&2
  exit "$node_setup_rc"
fi
if [ "$node_setup_actual" != "$node_setup_nvm_commit" ]; then
  echo "The managed Node installer has an unexpected revision. Move the managed bootstrap directory aside and retry; your existing Node installations are unchanged." >&2
  exit 1
fi
if git -C "$NVM_DIR" diff --quiet HEAD --; then
  :
else
  node_setup_rc=$?
  echo "Cannot use the modified or unreadable managed Node installer (git exit $node_setup_rc). Move the managed bootstrap directory aside and retry." >&2
  exit "$node_setup_rc"
fi

# nvm is a sourced shell dependency; it does not support nounset.
set +u
if source "$NVM_DIR/nvm.sh" --no-use >&2 && nvm install "$node_setup_version" >&2; then
  :
else
  node_setup_rc=$?
  echo "Cannot install .nvmrc's Node release (exit $node_setup_rc). Check the installer diagnostic above, network access and disk space, then retry." >&2
  exit "$node_setup_rc"
fi
node_setup_rc=0
node_setup_executable=$(nvm which "$node_setup_version") || node_setup_rc=$?
set -u
if [ "$node_setup_rc" -ne 0 ]; then
  echo "Cannot locate the installed Node release (exit $node_setup_rc). Check the installer diagnostic and retry." >&2
  exit "$node_setup_rc"
fi
if [ ! -x "$node_setup_executable" ]; then
  echo "The installed Node executable is missing. Move the incomplete bootstrap directory aside and retry." >&2
  exit 1
fi
node_setup_actual_version=$("$node_setup_executable" --version) || node_setup_rc=$?
if [ "$node_setup_rc" -ne 0 ]; then
  echo "Cannot run the installed Node executable (exit $node_setup_rc). Check platform compatibility and retry." >&2
  exit "$node_setup_rc"
fi
if [ "$node_setup_actual_version" != "v$node_setup_version" ]; then
  echo "The installed Node does not match .nvmrc. Repair or move the managed bootstrap directory aside, then retry." >&2
  exit 1
fi
dirname "$node_setup_executable"
