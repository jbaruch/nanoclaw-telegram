#!/usr/bin/env bash
# Install .nvmrc's exact Node release. stdout: JSON {"bin_dir":"..."};
# progress and actionable failures: stderr. Exit 0 on success, nonzero on
# failure. Does not edit the caller's shell profile.

node_setup_verify_cache() {
  local candidate="$1" expected="$2" actual rc=0
  if [ ! -f "$candidate/nvm.sh" ]; then
    echo "The managed Node installer is incomplete; downloading a verified replacement." >&2
    return 1
  fi
  actual=$(git -C "$candidate" rev-parse HEAD) || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "Cannot verify the managed Node installer (git exit $rc); downloading a verified replacement." >&2
    return "$rc"
  fi
  if [ "$actual" != "$expected" ]; then
    echo "The managed Node installer has an unexpected revision; a verified replacement is required." >&2
    return 1
  fi
  if git -C "$candidate" diff --quiet HEAD --; then
    return 0
  else
    rc=$?
    echo "The managed Node installer is modified or unreadable (git exit $rc); a verified replacement is required." >&2
    return "$rc"
  fi
}

node_setup_cleanup() {
  local primary=$? rc=0
  if [ -n "${node_setup_stage:-}" ]; then
    rm -rf "$node_setup_stage" || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "Cannot remove the partial Node download (rm exit $rc). Remove $node_setup_stage and retry." >&2
      if [ "$primary" -eq 0 ]; then exit "$rc"; fi
    fi
  fi
  return 0
}

install_node_main() {
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
  node_setup_stage=""
  if [ -d "$NVM_DIR" ] && node_setup_verify_cache "$NVM_DIR" "$node_setup_nvm_commit"; then
    :
  else
    node_setup_stage=$(mktemp -d "$node_setup_cache/nvm-download.XXXXXX")
    trap node_setup_cleanup EXIT
    if git clone --depth 1 --branch "$node_setup_nvm_tag" https://github.com/nvm-sh/nvm.git "$node_setup_stage" >&2; then
      :
    else
      node_setup_rc=$?
      echo "Cannot download the pinned Node installer (git exit $node_setup_rc). Check Git, network access and the bootstrap directory, then retry." >&2
      exit "$node_setup_rc"
    fi
    if node_setup_verify_cache "$node_setup_stage" "$node_setup_nvm_commit"; then
      :
    else
      node_setup_rc=$?
      echo "The downloaded installer failed verification. Check the pinned upstream release and retry; the existing cache is unchanged." >&2
      exit "$node_setup_rc"
    fi
    if [ -e "$NVM_DIR" ] || [ -L "$NVM_DIR" ]; then
      node_setup_backup=$(mktemp -d "$node_setup_cache/nvm-incomplete.XXXXXX")
      if mv "$NVM_DIR" "$node_setup_backup/original"; then
        echo "Preserved the incomplete managed cache at $node_setup_backup/original." >&2
      else
        node_setup_rc=$?
        echo "Cannot preserve the incomplete managed cache (mv exit $node_setup_rc). Check cache permissions and retry." >&2
        exit "$node_setup_rc"
      fi
    fi
    if mv "$node_setup_stage" "$NVM_DIR"; then
      node_setup_stage=""
      trap - EXIT
    else
      node_setup_rc=$?
      echo "Cannot install the verified Node cache (mv exit $node_setup_rc). Check cache permissions and retry; any earlier cache was preserved." >&2
      exit "$node_setup_rc"
    fi
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
  "$node_setup_executable" -e 'console.log(JSON.stringify({bin_dir:process.argv[1]}))' "$(dirname "$node_setup_executable")"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  install_node_main "$@"
fi
