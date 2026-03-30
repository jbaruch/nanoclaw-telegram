#!/usr/bin/env python3
"""
Deterministic system health checks for the heartbeat skill.

Runs all inline checks in parallel and outputs JSON:
  {
    "issues":  [{"check": "disk", "message": "87% used"}],
    "fixed":   [{"check": "logs", "message": "truncated app.log (120MB → 10k lines)"}],
    "ok":      ["disk", "logs", "sessions", "ipc", "containers"],
    "checked_at": "2026-03-29T05:00:00Z"
  }

Exit code 0 always. Caller decides what to do with issues.
"""

import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


LOGS_DIR       = Path('/workspace/group/logs')
PROJECTS_DIR   = Path('/home/node/.claude/projects')
IPC_INPUT_DIR  = Path('/workspace/ipc/input')
WORKSPACE_DIR  = '/workspace/group/'

DISK_WARN_PCT  = 80
DISK_CRIT_PCT  = 95
LOG_MAX_BYTES  = 50 * 1024 * 1024   # 50 MB
LOG_KEEP_LINES = 10_000
ORPHAN_LIMIT   = 5
SESSION_MAX_AGE_DAYS = 7
SESSION_KEEP_LATEST  = 5


def check_disk() -> dict:
    try:
        out = subprocess.check_output(
            ['df', '-h', WORKSPACE_DIR], text=True
        ).splitlines()
        fields = out[1].split()
        pct = int(fields[4].rstrip('%'))
        avail = fields[3]
        if pct >= DISK_CRIT_PCT:
            return {'status': 'critical', 'message': f'Disk {pct}% used ({avail} free) — CRITICAL'}
        if pct >= DISK_WARN_PCT:
            return {'status': 'issue', 'message': f'Disk {pct}% used ({avail} free)'}
        return {'status': 'ok'}
    except Exception as e:
        return {'status': 'issue', 'message': f'Disk check failed: {e}'}


def check_logs() -> dict:
    fixed = []
    try:
        if not LOGS_DIR.exists():
            return {'status': 'ok'}
        for f in LOGS_DIR.rglob('*'):
            if not f.is_file():
                continue
            size = f.stat().st_size
            if size > LOG_MAX_BYTES:
                size_mb = size / 1024 / 1024
                try:
                    lines = f.read_bytes().decode('utf-8', errors='replace').splitlines()
                    kept = lines[-LOG_KEEP_LINES:]
                    f.write_text('\n'.join(kept) + '\n')
                    fixed.append(f'truncated {f.name} ({size_mb:.0f}MB → {LOG_KEEP_LINES}k lines)')
                except Exception as e:
                    return {'status': 'issue', 'message': f'Failed to truncate {f.name}: {e}'}
        if fixed:
            return {'status': 'fixed', 'messages': fixed}
        return {'status': 'ok'}
    except Exception as e:
        return {'status': 'issue', 'message': f'Log check failed: {e}'}


def check_sessions() -> dict:
    fixed = []
    try:
        if not PROJECTS_DIR.exists():
            return {'status': 'ok'}
        from datetime import timedelta
        cutoff = datetime.now().timestamp() - SESSION_MAX_AGE_DAYS * 86400
        # Group by parent dir, keep latest N per group
        by_group: dict[Path, list[Path]] = {}
        for f in PROJECTS_DIR.rglob('*.jsonl'):
            by_group.setdefault(f.parent, []).append(f)
        for parent, files in by_group.items():
            files.sort(key=lambda f: f.stat().st_mtime)
            # Delete old files beyond keep count
            to_delete = files[:-SESSION_KEEP_LATEST] if len(files) > SESSION_KEEP_LATEST else []
            for f in to_delete:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    fixed.append(f.name)
        if fixed:
            return {'status': 'fixed', 'messages': [f'deleted {len(fixed)} stale sessions']}
        return {'status': 'ok'}
    except Exception as e:
        return {'status': 'issue', 'message': f'Session check failed: {e}'}


def check_ipc() -> dict:
    fixed = []
    try:
        if not IPC_INPUT_DIR.exists():
            return {'status': 'ok'}
        import time
        now = time.time()
        for f in IPC_INPUT_DIR.glob('_close'):
            age_min = (now - f.stat().st_mtime) / 60
            if age_min > 30:
                f.unlink()
                fixed.append(f'deleted stuck _close ({age_min:.0f}m old)')
        if fixed:
            return {'status': 'fixed', 'messages': fixed}
        return {'status': 'ok'}
    except Exception as e:
        return {'status': 'issue', 'message': f'IPC check failed: {e}'}


def check_containers() -> dict:
    try:
        result = subprocess.run(
            ['docker', 'ps', '-a',
             '--filter', 'name=nanoclaw-',
             '--filter', 'status=exited',
             '--format', '{{.Names}}'],
            capture_output=True, text=True, timeout=10
        )
        names = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        count = len(names)
        if count > ORPHAN_LIMIT:
            return {'status': 'issue', 'message': f'{count} orphaned nanoclaw containers'}
        return {'status': 'ok'}
    except FileNotFoundError:
        return {'status': 'ok'}   # docker not available in this env
    except Exception as e:
        return {'status': 'issue', 'message': f'Container check failed: {e}'}


def run_all() -> dict:
    checks = {
        'disk':       check_disk,
        'logs':       check_logs,
        'sessions':   check_sessions,
        'ipc':        check_ipc,
        'containers': check_containers,
    }
    issues = []
    fixed  = []
    ok     = []

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(fn): name for name, fn in checks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
            except Exception as e:
                result = {'status': 'issue', 'message': f'{name} check threw: {e}'}

            status = result.get('status', 'ok')
            if status == 'ok':
                ok.append(name)
            elif status == 'fixed':
                fixed.extend({'check': name, 'message': m}
                              for m in result.get('messages', [result.get('message', '')]))
                ok.append(name)
            elif status in ('issue', 'critical'):
                issues.append({'check': name, 'message': result.get('message', '')})

    return {
        'issues':     issues,
        'fixed':      fixed,
        'ok':         sorted(ok),
        'checked_at': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    }


if __name__ == '__main__':
    result = run_all()
    print(json.dumps(result, ensure_ascii=False, indent=2))
