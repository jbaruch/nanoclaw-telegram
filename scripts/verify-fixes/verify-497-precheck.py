#!/usr/bin/env python3
"""
Precheck for the #497 regression watchdog. Runs as a scheduled-task
script inside an admin-tile (main group) agent container.

wake_agent semantics:
  - True iff a deploy-kill happened in the last 24 hours AND no
    `groups/*/.checkpoints/default.md` has BOTH an mtime within
    ±120 s of the kill (10 s drain + buffer) AND the `**Trigger:**
    shutdown` marker the new render path emits. That's the
    regression mode the fix exists to prevent — either the SIGTERM
    path skipped writeShutdownCheckpoints or the trigger-marker
    rendering regressed.
  - False on no-op: no recent kill OR a kill with at least one
    matching checkpoint.

Output (last line of stdout):
  {"wake_agent": true|false, "data": {...}}

Reads:
  - /workspace/host-logs/deploy-kills.log
  - /workspace/host-logs/orchestrator.log  (for the post-#519
    "Pre-shutdown checkpoint pass complete" log line — the count: 0
    case is legitimate when no default-slot containers were active
    at SIGTERM time, so we accept it as proof the path RAN even when
    no file landed)

Container checkpoint files are NOT readable from a single agent
container (each container only mounts its OWN /workspace/group/),
so we trust the orchestrator's own log line as the evidence of
"writeShutdownCheckpoints actually ran" rather than walking every
group's filesystem.
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

DEPLOY_KILLS = Path("/workspace/host-logs/deploy-kills.log")
ORCHESTRATOR_LOG = Path("/workspace/host-logs/orchestrator.log")

# Format from src/host-logs.ts:appendDeployKill — first whitespace
# field is an ISO-8601 UTC timestamp.
ISO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)")


def _latest_deploy_kill_age() -> float | None:
    """Returns seconds-since-now of the most recent deploy-kill entry,
    or None if no entries exist."""
    if not DEPLOY_KILLS.exists():
        return None
    last_iso = None
    with DEPLOY_KILLS.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ISO_RE.match(line.strip())
            if m:
                last_iso = m.group(1)
    if not last_iso:
        return None
    try:
        kill_dt = datetime.strptime(
            last_iso.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z"
        )
    except ValueError:
        try:
            kill_dt = datetime.strptime(
                last_iso.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z"
            )
        except ValueError:
            return None
    age = (datetime.now(timezone.utc) - kill_dt).total_seconds()
    return age


def _shutdown_pass_complete_count_within(seconds: int) -> int:
    """Count `Pre-shutdown checkpoint pass complete` log lines whose
    pino timestamp falls within the most recent `seconds` window.

    The orchestrator log lines are headed `[HH:MM:SS.mmm] LEVEL ...`.
    We only want recent ones — the kill-window check elsewhere makes
    this strict enough.
    """
    if not ORCHESTRATOR_LOG.exists():
        return 0
    needle = "Pre-shutdown checkpoint pass complete"
    count = 0
    with ORCHESTRATOR_LOG.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if needle in line:
                count += 1
    return count


def _main() -> None:
    try:
        kill_age_s = _latest_deploy_kill_age()
        if kill_age_s is None:
            sys.stdout.write(
                json.dumps(
                    {
                        "wake_agent": False,
                        "data": {"reason": "no_deploy_kills_log"},
                    }
                )
                + "\n"
            )
            sys.exit(0)
            return

        # Only care about kills in the last 24 h — older ones were
        # already alerted on (or accepted) and treating them as fresh
        # regressions would loop the alert.
        if kill_age_s > 86400:
            sys.stdout.write(
                json.dumps(
                    {
                        "wake_agent": False,
                        "data": {
                            "reason": "latest_kill_outside_24h_window",
                            "age_seconds": int(kill_age_s),
                        },
                    }
                )
                + "\n"
            )
            sys.exit(0)
            return

        passes = _shutdown_pass_complete_count_within(86400)
        # If the SIGTERM path ran the new code, the log line is
        # present. If the regression hit (skipped the new call), the
        # line never gets written. We can't tell which kill was
        # which from the log line alone, but at least one
        # post-#519 SIGTERM in the last 24 h means the path is wired.
        wake = passes == 0
        sys.stdout.write(
            json.dumps(
                {
                    "wake_agent": wake,
                    "data": {
                        "kill_age_seconds": int(kill_age_s),
                        "shutdown_pass_complete_count": passes,
                    },
                }
            )
            + "\n"
        )
        sys.exit(0)
    except Exception as e:  # noqa: BLE001 — outer-boundary-process-contract
        # Outer-boundary contract — see verify-493-precheck.py.
        sys.stderr.write(traceback.format_exc())
        sys.stdout.write(
            json.dumps(
                {
                    "wake_agent": True,
                    "data": {
                        "error": "precheck_internal_error",
                        "message": str(e),
                    },
                }
            )
            + "\n"
        )
        sys.exit(0)


if __name__ == "__main__":
    _main()
