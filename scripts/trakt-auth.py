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


def _load_credentials() -> tuple[str, str]:
    client_id = os.environ.get("TRAKT_CLIENT_ID")
    client_secret = os.environ.get("TRAKT_CLIENT_SECRET")
    if client_id and client_secret:
        return client_id, client_secret

    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TRAKT_CLIENT_ID="):
                    client_id = line.split("=", 1)[1]
                elif line.startswith("TRAKT_CLIENT_SECRET="):
                    client_secret = line.split("=", 1)[1]

    if not client_id or not client_secret:
        print(
            "ERROR: TRAKT_CLIENT_ID and TRAKT_CLIENT_SECRET must be in .env or environment"
        )
        sys.exit(1)
    return client_id, client_secret


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
            if e.code == 400:
                print("  Still waiting...")
                continue
            elif e.code == 410:
                print("  ERROR: Code expired. Run again.")
                sys.exit(1)
            elif e.code == 418:
                print("  ERROR: User denied access")
                sys.exit(1)
            elif e.code == 429:
                print("  Polling too fast, slowing down...")
                interval += 1
                continue
            raise

    access_token = token_resp["access_token"]
    refresh_token = token_resp["refresh_token"]

    print("\n  Authenticated successfully!")

    env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    with open(env_path, "a") as f:
        f.write(f"\nTRAKT_ACCESS_TOKEN={access_token}\n")
        f.write(f"TRAKT_REFRESH_TOKEN={refresh_token}\n")

    print(f"  Tokens saved to {env_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
