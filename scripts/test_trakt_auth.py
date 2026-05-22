"""
stdlib-unittest contract test for `scripts/trakt-auth.py`. Run with:
  python3 -m unittest scripts.test_trakt_auth

No pytest dependency — the host repo is TS-first and `trakt-auth.py`
is the only Python script. Keeping the test stdlib-only avoids
introducing a Python testing toolchain for one file.
"""
import importlib.util
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_SCRIPT_PATH = Path(__file__).resolve().parent / "trakt-auth.py"


def _load_script():
    # The script's filename contains a hyphen, so it isn't a valid
    # Python identifier for normal import. Load it via spec so the
    # test can target its functions directly.
    spec = importlib.util.spec_from_file_location("trakt_auth", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildHeadersContractTest(unittest.TestCase):
    """`_build_headers` returns the five Trakt-required headers plus
    a browser-shaped User-Agent. Cloudflare in front of api.trakt.tv
    flags short custom UAs, so the contract this test pins is
    "browser-shaped UA + Trakt API headers" — not a specific Chrome
    version string."""

    def setUp(self):
        self.module = _load_script()

    def test_returns_browser_shaped_user_agent(self):
        headers = self.module._build_headers("dummy-client-id")
        ua = headers["User-Agent"]
        self.assertTrue(
            ua.startswith("Mozilla/5.0"),
            f"User-Agent must look like a real browser, got: {ua!r}",
        )
        self.assertIn("Chrome/", ua)
        self.assertIn("Safari/", ua)

    def test_carries_required_trakt_headers(self):
        headers = self.module._build_headers("cid-123")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["trakt-api-version"], "2")
        self.assertEqual(headers["trakt-api-key"], "cid-123")

    def test_browser_ua_module_constant_matches_build_headers(self):
        """`BROWSER_UA` and `_build_headers(...)["User-Agent"]` must
        not drift. The whole point of the constant is one source of
        truth across the device-code and token-poll requests."""
        headers = self.module._build_headers("x")
        self.assertEqual(headers["User-Agent"], self.module.BROWSER_UA)


class DeviceCodeRequestUsesBrowserUA(unittest.TestCase):
    """End-to-end-ish: monkeypatch urlopen + sleep, run main(), and
    verify the FIRST Request (the device-code call) carries
    `BROWSER_UA` in its headers. We stop the flow after that — the
    token-poll loop uses the same `headers` dict by construction
    (see the refactor: `headers = _build_headers(client_id)` is
    bound once and reused), so a single capture pins both paths."""

    def setUp(self):
        self.module = _load_script()

    def test_first_request_user_agent_matches_browser_ua(self):
        captured = {}

        class _FakeResp:
            def __init__(self, payload):
                self._payload = json.dumps(payload).encode()

            def read(self):
                return self._payload

        def _fake_urlopen(req, *args, **kwargs):
            if "request" not in captured:
                captured["request"] = req
            # Returning a device-code-shaped response makes main()
            # proceed to the polling loop; we cut it off there via
            # SystemExit from the 410 branch.
            return _FakeResp(
                {
                    "verification_url": "https://example.invalid",
                    "user_code": "TEST",
                    "expires_in": 10,
                    "device_code": "dc",
                    "interval": 0,
                }
            )

        def _fake_urlopen_then_410(req, *args, **kwargs):
            # First call: device-code success. Second call: 410 so
            # main() exits without us having to construct a full
            # token response.
            if "request" not in captured:
                captured["request"] = req
                return _FakeResp(
                    {
                        "verification_url": "https://example.invalid",
                        "user_code": "TEST",
                        "expires_in": 10,
                        "device_code": "dc",
                        "interval": 0,
                    }
                )
            import urllib.error

            raise urllib.error.HTTPError(
                req.full_url, 410, "Gone", hdrs=None, fp=io.BytesIO(b"")
            )

        env = {
            "TRAKT_CLIENT_ID": "cid",
            "TRAKT_CLIENT_SECRET": "secret",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch(
            "urllib.request.urlopen", side_effect=_fake_urlopen_then_410
        ), mock.patch.object(self.module.time, "sleep", lambda _s: None), mock.patch(
            "sys.stdout", new_callable=io.StringIO
        ):
            with self.assertRaises(SystemExit):
                self.module.main()

        req = captured["request"]
        # `urllib.request.Request` normalizes header names by
        # title-casing the first letter, so `User-Agent` arrives as
        # `User-agent`.
        self.assertEqual(req.headers["User-agent"], self.module.BROWSER_UA)
        self.assertEqual(req.headers["Trakt-api-key"], "cid")


class PersistTokensTest(unittest.TestCase):
    """`_persist_tokens` rewrites .env in place. Contract:
    - Inode stable across the rewrite (docker bind-mount safety).
    - Stacked pre-existing TRAKT_*_TOKEN= lines collapse to exactly
      one of each key (first match wins, subsequent dropped).
    - Other .env vars preserved verbatim.
    - Append-only when no prior key exists."""

    def setUp(self):
        self.module = _load_script()
        # tempdir + tempfile manually since this is stdlib-only.
        import tempfile

        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.env_path = os.path.join(self._dir.name, ".env")

    def _read_lines(self):
        with open(self.env_path) as f:
            return f.read().splitlines()

    def test_rewrites_in_place_preserving_inode(self):
        with open(self.env_path, "w") as f:
            f.write("TRAKT_ACCESS_TOKEN=old\nTRAKT_REFRESH_TOKEN=old-r\n")
        inode_before = os.stat(self.env_path).st_ino
        self.module._persist_tokens(self.env_path, "new", "new-r")
        inode_after = os.stat(self.env_path).st_ino
        self.assertEqual(
            inode_before,
            inode_after,
            "inode swap would break docker bind-mounts",
        )
        lines = self._read_lines()
        self.assertIn("TRAKT_ACCESS_TOKEN=new", lines)
        self.assertIn("TRAKT_REFRESH_TOKEN=new-r", lines)

    def test_collapses_stacked_pre_existing_duplicates(self):
        with open(self.env_path, "w") as f:
            f.write(
                "OTHER=keep\n"
                "TRAKT_ACCESS_TOKEN=stale-1\n"
                "TRAKT_REFRESH_TOKEN=stale-1\n"
                "MIDDLE=between\n"
                "TRAKT_ACCESS_TOKEN=stale-2\n"
                "TRAKT_REFRESH_TOKEN=stale-2\n"
                "TRAKT_ACCESS_TOKEN=stale-3\n"
                "TAIL=tail\n"
            )
        self.module._persist_tokens(self.env_path, "fresh", "fresh-r")
        lines = self._read_lines()
        access = [line for line in lines if line.startswith("TRAKT_ACCESS_TOKEN=")]
        refresh = [line for line in lines if line.startswith("TRAKT_REFRESH_TOKEN=")]
        self.assertEqual(access, ["TRAKT_ACCESS_TOKEN=fresh"], lines)
        self.assertEqual(refresh, ["TRAKT_REFRESH_TOKEN=fresh-r"], lines)
        # Non-Trakt vars preserved verbatim.
        self.assertIn("OTHER=keep", lines)
        self.assertIn("MIDDLE=between", lines)
        self.assertIn("TAIL=tail", lines)

    def test_appends_when_no_prior_keys(self):
        with open(self.env_path, "w") as f:
            f.write("OTHER=keep\n")
        self.module._persist_tokens(self.env_path, "new", "new-r")
        lines = self._read_lines()
        self.assertIn("OTHER=keep", lines)
        self.assertIn("TRAKT_ACCESS_TOKEN=new", lines)
        self.assertIn("TRAKT_REFRESH_TOKEN=new-r", lines)

    def test_idempotent_across_multiple_runs(self):
        """`jbaruch/coding-policy: file-hygiene` — Scripts should be
        safe to run multiple times. Three sequential _persist_tokens
        calls must leave exactly one of each key."""
        for i in range(3):
            self.module._persist_tokens(self.env_path, f"a-{i}", f"r-{i}")
        lines = self._read_lines()
        access = [line for line in lines if line.startswith("TRAKT_ACCESS_TOKEN=")]
        refresh = [line for line in lines if line.startswith("TRAKT_REFRESH_TOKEN=")]
        self.assertEqual(access, ["TRAKT_ACCESS_TOKEN=a-2"])
        self.assertEqual(refresh, ["TRAKT_REFRESH_TOKEN=r-2"])


if __name__ == "__main__":
    sys.exit(unittest.main())
