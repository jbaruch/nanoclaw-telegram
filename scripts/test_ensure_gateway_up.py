"""
stdlib-unittest contract test for scripts/ensure-gateway-up.sh — the
deploy.sh step-8 verify-and-re-up loop for the nanoclaw-litellm gateway.

Run with:
  python3 -m unittest scripts.test_ensure_gateway_up

The script's docker calls (`compose up -d` / `compose ps -q` / `inspect`)
are the genuinely-unhostable layer; its retry CONTROL FLOW is
deterministic and is what we test here, with `docker` stubbed on PATH so
no real Docker daemon or gateway is needed (coding-policy:
testing-standards platform-bound carve-out). A regression either
(a) fire-and-forgets `up -d` without confirming the container reached
`running` — the exact gap that let a SIGKILLed gateway stay down while
the orchestrator bypassed to anthropic-direct (see CHANGELOG) — or
(b) makes a still-down gateway abort the deploy, when the bypass means it
must stay non-fatal.

The stub `docker`:
  compose ps -q <svc>  -> prints a fake container id (container exists)
  compose up -d        -> appends a line to $STUB_DIR/up.log
  inspect ... <cid>    -> pops the next line of $STUB_DIR/states as the
                          container State.Status (default "exited" once
                          the file is exhausted)
so a test drives the running/not-running sequence purely via `states`.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parent / "ensure-gateway-up.sh"

_STUB_DOCKER = r"""#!/usr/bin/env bash
set -u
if [ "${1:-}" = "compose" ]; then
    case "${2:-}" in
        ps) echo "fakecid000" ;;
        up) echo up >> "$STUB_DIR/up.log" ;;
    esac
    exit 0
fi
if [ "${1:-}" = "inspect" ]; then
    states="$STUB_DIR/states"
    if [ -s "$states" ]; then
        head -1 "$states"
        tail -n +2 "$states" > "$states.tmp" && mv "$states.tmp" "$states"
    else
        echo exited
    fi
    exit 0
fi
exit 0
"""


def _run(states: list[str], max_attempts: int = 3):
    """Run the script with a stubbed docker; return (proc, up_call_count)."""
    with tempfile.TemporaryDirectory() as stub_dir, tempfile.TemporaryDirectory() as project_dir:
        docker = Path(stub_dir) / "docker"
        docker.write_text(_STUB_DOCKER)
        docker.chmod(
            docker.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH
        )
        (Path(stub_dir) / "states").write_text(
            ("\n".join(states) + "\n") if states else ""
        )
        env = dict(os.environ)
        env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
        env["STUB_DIR"] = stub_dir
        env["GATEWAY_SETTLE_SECS"] = "0"
        env["GATEWAY_MAX_ATTEMPTS"] = str(max_attempts)
        proc = subprocess.run(
            ["bash", str(_SCRIPT), project_dir],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env=env,
        )
        up_log = Path(stub_dir) / "up.log"
        up_calls = len(up_log.read_text().splitlines()) if up_log.exists() else 0
        return proc, up_calls


class EnsureGatewayUpTests(unittest.TestCase):
    def test_running_first_check_no_retry(self) -> None:
        # Healthy gateway: one `up -d`, the running-check passes, done.
        proc, up_calls = _run(["running"])
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(up_calls, 1)
        self.assertIn("ok — gateway running", proc.stdout)
        self.assertNotIn("WARNING", proc.stderr)

    def test_recovers_after_one_failed_start(self) -> None:
        # The deploy-race shape: the first start dies, the re-up sticks.
        proc, up_calls = _run(["exited", "running"])
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(up_calls, 2)
        self.assertIn("re-upping", proc.stdout)
        self.assertIn("ok — gateway running", proc.stdout)
        self.assertNotIn("WARNING", proc.stderr)

    def test_warns_non_fatal_when_never_running(self) -> None:
        # Gateway never comes up: exactly max_attempts starts, then a loud
        # WARNING — but exit 0, because the orchestrator's anthropic-direct
        # bypass keeps serving and a still-down gateway must not abort the
        # deploy.
        proc, up_calls = _run([], max_attempts=3)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(up_calls, 3)
        self.assertIn("WARNING", proc.stderr)
        self.assertIn("anthropic-direct", proc.stderr)


if __name__ == "__main__":
    unittest.main()
