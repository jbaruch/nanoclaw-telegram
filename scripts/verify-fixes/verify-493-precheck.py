#!/usr/bin/env python3
"""
Precheck for the #493 regression watchdog. Runs as a scheduled-task
script inside an admin-tile (main group) agent container.

wake_agent semantics:
  - True iff non-main inbound traffic happened in the last 60 minutes
    AND zero `tier:'classifier'` records landed in usage.jsonl over
    the same window. That's the regression mode the fix exists to
    prevent — either the haiku-classifier gate stopped firing or
    the #493 emit hook silently regressed.
  - False on no-op: traffic is fine and classifier records are
    landing, OR no non-main traffic at all (no signal either way).

Output (last line of stdout):
  {"wake_agent": true|false, "data": {...}}

The agent-runner reads the last JSON line of stdout per
`coding-policy: script-delegation`. Always exit 0 so a parser
failure doesn't silently disable the watchdog (the outer-boundary
contract per `coding-policy: error-handling`).

Reads:
  - /workspace/host-logs/usage.jsonl    (RO, mounted in #522)
  - /workspace/store/messages.db        (RO for trusted, RW for main)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

USAGE_LOG = Path("/workspace/host-logs/usage.jsonl")
STORE_DB = Path("/workspace/store/messages.db")


def _classifier_count(window_start_iso: str) -> int:
    """Count usage.jsonl records with tier:'classifier' since the cutoff."""
    if not USAGE_LOG.exists():
        return 0
    count = 0
    with USAGE_LOG.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or '"tier":"classifier"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("ts", "")
            if isinstance(ts, str) and ts >= window_start_iso:
                count += 1
    return count


def _inbound_non_main_count(window_start_iso: str) -> int:
    """Count inbound (not-from-me) messages on non-main groups since the cutoff."""
    if not STORE_DB.exists():
        return 0
    conn = sqlite3.connect(f"file:{STORE_DB}?mode=ro", uri=True)
    try:
        cur = conn.execute(
            """
            SELECT COUNT(*) FROM messages m
            JOIN registered_groups g ON g.jid = m.chat_jid
            WHERE m.timestamp >= ?
              AND COALESCE(g.is_main, 0) = 0
              AND COALESCE(m.is_from_me, 0) = 0
            """,
            (window_start_iso,),
        )
        return cur.fetchone()[0] or 0
    finally:
        conn.close()


def _main() -> None:
    try:
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=1)
        window_start_iso = window_start.strftime("%Y-%m-%dT%H:%M:%SZ")

        classifier = _classifier_count(window_start_iso)
        inbound = _inbound_non_main_count(window_start_iso)

        # Regression: traffic but no classifier emits.
        wake = inbound > 0 and classifier == 0

        sys.stdout.write(
            json.dumps(
                {
                    "wake_agent": wake,
                    "data": {
                        "window_start": window_start_iso,
                        "classifier_count": classifier,
                        "inbound_non_main_count": inbound,
                    },
                }
            )
            + "\n"
        )
        sys.exit(0)
    except Exception as e:  # noqa: BLE001 — outer-boundary-process-contract
        # Outer-boundary process contract per `coding-policy:
        # error-handling`: agent-runner reads non-zero exit / bad JSON
        # as wake_agent=false. A propagated TypeError from a
        # programmer bug here would silently disable the watchdog —
        # exactly the failure mode this catch exists to prevent.
        # Stamp wake_agent=true so a regressed precheck wakes the
        # agent (it'll see the error in `data` and report).
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
