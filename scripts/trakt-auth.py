#!/usr/bin/env python3
"""
Trakt.tv device auth flow. Run once to get OAuth tokens.
Saves tokens to .env as TRAKT_ACCESS_TOKEN and TRAKT_REFRESH_TOKEN.

Usage: Run on the NAS via docker exec:
  docker exec nanoclaw python3 /app/scripts/trakt-auth.py
"""
import json, os, sys, time, urllib.request

# Browser-shaped User-Agent. Cloudflare in front of api.trakt.tv
# flags short custom UAs; `NanoClaw/1.0` was being intermittently
# blocked. Match the UA used by the runtime fetcher so reauth and
# fetch share one Cloudflare fingerprint.
BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

CLIENT_ID = os.environ.get("TRAKT_CLIENT_ID")
CLIENT_SECRET = os.environ.get("TRAKT_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    # Try reading from .env
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TRAKT_CLIENT_ID="):
                    CLIENT_ID = line.split("=", 1)[1]
                elif line.startswith("TRAKT_CLIENT_SECRET="):
                    CLIENT_SECRET = line.split("=", 1)[1]

if not CLIENT_ID or not CLIENT_SECRET:
    print("ERROR: TRAKT_CLIENT_ID and TRAKT_CLIENT_SECRET must be in .env or environment")
    sys.exit(1)

# Step 1: Request device code
req = urllib.request.Request(
    "https://api.trakt.tv/oauth/device/code",
    data=json.dumps({"client_id": CLIENT_ID}).encode(),
    headers={
        "Content-Type": "application/json",
        "trakt-api-version": "2",
        "trakt-api-key": CLIENT_ID,
        "User-Agent": BROWSER_UA,
    },
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
            data=json.dumps({
                "code": device_code,
                "client_id": CLIENT_ID,
                "client_secret": CLIENT_SECRET,
            }).encode(),
            headers={
        "Content-Type": "application/json",
        "trakt-api-version": "2",
        "trakt-api-key": CLIENT_ID,
        "User-Agent": BROWSER_UA,
    },
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

print(f"\n  Authenticated successfully!")
print(f"  Access token: {access_token[:20]}...")

# Append to .env
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
with open(env_path, "a") as f:
    f.write(f"\nTRAKT_ACCESS_TOKEN={access_token}\n")
    f.write(f"TRAKT_REFRESH_TOKEN={refresh_token}\n")

print(f"  Tokens saved to {env_path}")
