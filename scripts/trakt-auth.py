#!/usr/bin/env python3
"""
Trakt.tv device auth flow. Run once to get OAuth tokens.
Saves tokens to .env as TRAKT_ACCESS_TOKEN and TRAKT_REFRESH_TOKEN.

Usage: Run on the NAS via docker exec:
  docker exec nanoclaw python3 /app/scripts/trakt-auth.py
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

# Browser-shaped User-Agent. Cloudflare in front of api.trakt.tv
# flags short custom UAs; `NanoClaw/1.0` was being intermittently
# blocked. Matches the UA used by the runtime fetcher in
# `jbaruch/nanoclaw-admin` (`skills/trakt-watch-history/scripts/
# trakt-watch-history.py`, see PR #293) so reauth and fetch share
# one Cloudflare fingerprint. The Chrome major version isn't
# load-bearing beyond "looks like a real recent browser"; bump if
# Cloudflare tightens.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/141.0.0.0 Safari/537.36"
)


def _build_headers(client_id: str) -> dict:
    return {
        "Content-Type": "application/json",
        "trakt-api-version": "2",
        "trakt-api-key": client_id,
        "User-Agent": BROWSER_UA,
    }


def _env_path() -> str:
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )


def _load_credentials() -> tuple[str, str]:
    client_id = os.environ.get("TRAKT_CLIENT_ID")
    client_secret = os.environ.get("TRAKT_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret

    env_path = _env_path()
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TRAKT_CLIENT_ID="):
                    client_id = line.split("=", 1)[1]
                elif line.startswith("TRAKT_CLIENT_SECRET="):
                    client_secret = line.split("=", 1)[1]

    if not client_id or not client_secret:
        # stderr per `jbaruch/coding-policy: file-hygiene`. stdout is
        # reserved for the interactive auth flow's verification URL +
        # user code, which the operator copies into a browser.
        print(
            "ERROR: TRAKT_CLIENT_ID and TRAKT_CLIENT_SECRET must be in .env or environment",
            file=sys.stderr,
        )
        sys.exit(1)
    return client_id, client_secret


def _persist_tokens(env_path: str, access_token: str, refresh_token: str) -> None:
    """Rewrite .env in place, preserving the inode so a docker
    bind-mount of this file continues to see the new content.
    Replaces existing TRAKT_ACCESS_TOKEN / TRAKT_REFRESH_TOKEN lines
    if present (first match wins, subsequent stacked duplicates are
    dropped — older runs of this script appended on every invocation
    rather than replacing, leaving stacked tokens in .env); appends
    both if absent. All other lines preserved verbatim.

    Same rewrite shape as the runtime watch-history script's
    `_persist_tokens_to_env` (jbaruch/nanoclaw-admin) so the two
    writers don't drift on .env layout.

    NB: temp-file + rename would swap the inode and break the bind
    mount — bind mounts attach to the inode, not the path, and the
    renamed file ends up unmounted from the container. `open("w")`
    truncates the existing file in place, keeping the inode."""
    try:
        with open(env_path, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []

    seen_access = False
    seen_refresh = False
    out_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("TRAKT_ACCESS_TOKEN="):
            if not seen_access:
                out_lines.append(f"TRAKT_ACCESS_TOKEN={access_token}\n")
                seen_access = True
        elif stripped.startswith("TRAKT_REFRESH_TOKEN="):
            if not seen_refresh:
                out_lines.append(f"TRAKT_REFRESH_TOKEN={refresh_token}\n")
                seen_refresh = True
        else:
            out_lines.append(line)

    if not seen_access:
        out_lines.append(f"TRAKT_ACCESS_TOKEN={access_token}\n")
    if not seen_refresh:
        out_lines.append(f"TRAKT_REFRESH_TOKEN={refresh_token}\n")

    with open(env_path, "w") as f:
        f.writelines(out_lines)


def main() -> int:
    client_id, client_secret = _load_credentials()
    headers = _build_headers(client_id)

    # Step 1: Request device code
    req = urllib.request.Request(
        "https://api.trakt.tv/oauth/device/code",
        data=json.dumps({"client_id": client_id}).encode(),
        headers=headers,
    )
    resp = json.loads(urllib.request.urlopen(req).read())

    print(f"\n  Go to: {resp['verification_url']}")
    print(f"  Enter code: {resp['user_code']}\n")
    print(f"  Waiting for confirmation (expires in {resp['expires_in']}s)...")

    # Step 2: Poll for token
    device_code = resp["device_code"]
    interval = resp["interval"]

    while True:
        time.sleep(interval)
        try:
            req = urllib.request.Request(
                "https://api.trakt.tv/oauth/device/token",
                data=json.dumps(
                    {
                        "code": device_code,
                        "client_id": client_id,
                        "client_secret": client_secret,
                    }
                ).encode(),
                headers=headers,
            )
            token_resp = json.loads(urllib.request.urlopen(req).read())
            break
        except urllib.error.HTTPError as e:
            # `400 (still waiting)` and `429 (polling too fast)` are
            # expected on stdout — they're part of the interactive
            # flow. Terminal errors (`410 expired`, `418 denied`) and
            # the diagnostic above the rate-limit retry go to stderr
            # per file-hygiene.
            if e.code == 400:
                print("  Still waiting...")
                continue
            elif e.code == 410:
                print(
                    "  ERROR: Device code expired (10-minute window elapsed). "
                    "Re-run this script to start a fresh device-code flow.",
                    file=sys.stderr,
                )
                sys.exit(1)
            elif e.code == 418:
                print(
                    "  ERROR: Authorization denied. Re-run this script and "
                    "approve the Trakt authorization prompt at the URL "
                    "displayed above.",
                    file=sys.stderr,
                )
                sys.exit(1)
            elif e.code == 429:
                print("  Polling too fast, slowing down...", file=sys.stderr)
                interval += 1
                continue
            raise

    access_token = token_resp["access_token"]
    refresh_token = token_resp["refresh_token"]

    print("\n  Authenticated successfully!")

    env_path = _env_path()
    _persist_tokens(env_path, access_token, refresh_token)

    print(f"  Tokens saved to {env_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
