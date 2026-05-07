#!/usr/bin/env python3
"""
Precheck for the #451 IPC regression watchdog. Runs as a
scheduled-task script inside an admin-tile (main group) agent
container.

wake_agent semantics:
  - True iff the `list_learned_triggers` IPC silently regressed:
    no result file appeared within 10 s, the response shape lacks
    a `groups` array, or the handler returned an `error`.
  - False on healthy roundtrip.

Output (last line of stdout):
  {"wake_agent": true|false, "data": {...}}

Reads/writes:
  - Drops a request file at
    /workspace/ipc/<main-folder>/tasks/req-<id>.json
  - Reads the result at
    /workspace/ipc/<main-folder>/input-default/_script_result_<id>.json
  - Cleans up both files on completion so disk usage stays bounded.

The `<main-folder>` is resolved via /workspace/store/messages.db
(read-only). The script doesn't hardcode `telegram_swarm` so the
verification keeps working if the main-group folder is renamed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import traceback
import uuid
from pathlib import Path

STORE_DB = Path("/workspace/store/messages.db")
IPC_ROOT = Path("/workspace/ipc")


def _resolve_main_folder() -> str | None:
    if not STORE_DB.exists():
        return None
    conn = sqlite3.connect(f"file:{STORE_DB}?mode=ro", uri=True)
    try:
        row = conn.execute(
            "SELECT folder FROM registered_groups WHERE is_main = 1 LIMIT 1"
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _main() -> None:
    try:
        main_folder = _resolve_main_folder()
        if main_folder is None:
            sys.stdout.write(
                json.dumps(
                    {
                        "wake_agent": False,
                        "data": {"reason": "no_main_group_registered"},
                    }
                )
                + "\n"
            )
            sys.exit(0)
            return

        req_id = "verify-451-" + uuid.uuid4().hex[:12]
        tasks_dir = IPC_ROOT / main_folder / "tasks"
        result_dir = IPC_ROOT / main_folder / "input-default"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        req_file = tasks_dir / f"req-{req_id}.json"
        result_file = result_dir / f"_script_result_{req_id}.json"

        req_file.write_text(
            json.dumps(
                {"type": "list_learned_triggers", "requestId": req_id}
            ),
            encoding="utf-8",
        )

        deadline = time.time() + 10.0
        while time.time() < deadline:
            if result_file.exists():
                break
            time.sleep(0.5)

        wake = False
        data: dict = {"req_id": req_id}

        if not result_file.exists():
            wake = True
            data["reason"] = "no_result_file_within_10s"
        else:
            try:
                payload = json.loads(result_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError as parse_err:
                wake = True
                data["reason"] = "result_not_json"
                data["parse_error"] = str(parse_err)
                payload = None

            if isinstance(payload, dict):
                if "error" in payload:
                    wake = True
                    data["reason"] = "handler_returned_error"
                    data["error"] = payload["error"]
                else:
                    stdout_str = payload.get("stdout")
                    inner = None
                    if isinstance(stdout_str, str):
                        try:
                            inner = json.loads(stdout_str)
                        except json.JSONDecodeError:
                            inner = None
                    if not isinstance(inner, dict) or not isinstance(
                        inner.get("groups"), list
                    ):
                        wake = True
                        data["reason"] = "response_shape_invalid"
                    else:
                        data["group_count"] = len(inner["groups"])
                        data["learned_total"] = sum(
                            len(g.get("learned", []))
                            for g in inner["groups"]
                            if isinstance(g, dict)
                        )

        # Cleanup so disk doesn't accumulate request/result pairs.
        try:
            if req_file.exists():
                req_file.unlink()
        except OSError:
            pass
        try:
            if result_file.exists():
                result_file.unlink()
        except OSError:
            pass

        sys.stdout.write(
            json.dumps({"wake_agent": wake, "data": data}) + "\n"
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
