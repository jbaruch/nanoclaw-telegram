"""stdlib-unittest tests for the pure helpers in backup.py.

Run with:
  python3 -m unittest container.audible-backup.test_backup

The download path itself shells out to audible-cli + ffmpeg, which the
testing-standards rule says we don't try to mock; instead we cover the
deterministic pieces extracted out of `download_and_decrypt` so a
regression in the discovery / voucher-parse / classifier logic surfaces
without needing an Audible account or pinned binary fixtures.

The lifted backup.py module isn't an importable package (the directory
has a `-` in its name), so the test file injects its directory onto
sys.path and imports the module directly. Same shape the other
sidecar-adjacent tests in this repo use.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import backup  # noqa: E402


class ParseSkippedPathsTest(unittest.TestCase):
    """`parse_skipped_paths` walks audible-cli's combined stdout/stderr."""

    def test_extracts_paths_from_skip_lines(self):
        stdout = (
            "File /library/tmp_download/Mr._Mulliner_Speaking-AAX_22_64.aaxc "
            "already exists. Skip download.\n"
            "File /library/tmp_download/Mr._Mulliner_Speaking-AAX_22_64.voucher "
            "already exists. Skip download.\n"
            "File /library/tmp_download/Mr._Mulliner_Speaking-chapters.json "
            "already exists. Skip saving chapters\n"
        )
        paths = backup.parse_skipped_paths(stdout)
        # The "Skip saving chapters" line ALSO matches because the
        # regex anchors on the verbatim prefix; both shapes are valid
        # signal that the file is present on disk. Operator-facing
        # consequence is fine — we filter to `is_file()` after.
        self.assertEqual(
            [str(p) for p in paths],
            [
                "/library/tmp_download/Mr._Mulliner_Speaking-AAX_22_64.aaxc",
                "/library/tmp_download/Mr._Mulliner_Speaking-AAX_22_64.voucher",
                "/library/tmp_download/Mr._Mulliner_Speaking-chapters.json",
            ],
        )

    def test_empty_when_no_skip_lines(self):
        self.assertEqual(backup.parse_skipped_paths(""), [])
        self.assertEqual(
            backup.parse_skipped_paths("Downloading: Uncle Dynamite\nDone."),
            [],
        )

    def test_paths_with_literal_spaces_are_not_matched(self):
        # The regex template is `File (\S+) already exists` — it anchors
        # on the literal trailing prose, so a path containing a literal
        # space breaks the match entirely (the non-greedy `\S+` stops at
        # the space, then the trailing ` already exists` doesn't line up
        # against the next token). audible-cli sanitises spaces in
        # titles to underscores by default, so this is an acceptable
        # gap; documenting it here so a future "let's just match any
        # path shape" rewrite doesn't reintroduce false captures.
        paths = backup.parse_skipped_paths(
            "File /library/tmp_download/With Space.aax already exists. Skip download.\n"
        )
        self.assertEqual(paths, [])


class ClassifyArtifactsTest(unittest.TestCase):
    """`classify_artifacts` buckets touched files by role."""

    def test_aax_only(self):
        touched = [
            Path("/d/Red_Rising-LC_128_44100_stereo.aax"),
            Path("/d/Red_Rising-chapters.json"),
            Path("/d/Red_Rising_(500).jpg"),
        ]
        artifacts = backup.classify_artifacts(touched)
        self.assertEqual(
            artifacts["audio_file"],
            Path("/d/Red_Rising-LC_128_44100_stereo.aax"),
        )
        self.assertEqual(artifacts["source_kind"], "aax")
        self.assertIsNone(artifacts["voucher"])
        self.assertEqual(artifacts["cover"], Path("/d/Red_Rising_(500).jpg"))

    def test_aaxc_with_voucher(self):
        touched = [
            Path("/d/Mulliner-AAX_22_64.aaxc"),
            Path("/d/Mulliner-AAX_22_64.voucher"),
            Path("/d/Mulliner-chapters.json"),
            Path("/d/Mulliner_(500).png"),
        ]
        artifacts = backup.classify_artifacts(touched)
        self.assertEqual(artifacts["audio_file"], Path("/d/Mulliner-AAX_22_64.aaxc"))
        self.assertEqual(artifacts["source_kind"], "aaxc")
        self.assertEqual(artifacts["voucher"], Path("/d/Mulliner-AAX_22_64.voucher"))
        self.assertEqual(artifacts["cover"], Path("/d/Mulliner_(500).png"))

    def test_aaxc_preferred_over_aax(self):
        # audible-cli has been observed to deliver both formats for the
        # same title; AAXC ranks higher because its voucher ships in the
        # download (no per-account activation-bytes round-trip required).
        touched = [
            Path("/d/Mixed-LC_64_22050_stereo.aax"),
            Path("/d/Mixed-AAX_22_64.aaxc"),
            Path("/d/Mixed-AAX_22_64.voucher"),
        ]
        artifacts = backup.classify_artifacts(touched)
        self.assertEqual(artifacts["audio_file"], Path("/d/Mixed-AAX_22_64.aaxc"))
        self.assertEqual(artifacts["source_kind"], "aaxc")

    def test_no_audio_returns_none(self):
        touched = [
            Path("/d/Galaxys_Edge_(500).jpg"),
            Path("/d/Galaxys_Edge-chapters.json"),
        ]
        artifacts = backup.classify_artifacts(touched)
        self.assertIsNone(artifacts["audio_file"])
        self.assertIsNone(artifacts["source_kind"])
        self.assertIsNone(artifacts["voucher"])
        # Cover still surfaces even when no audio — the diagnostic log
        # in download_and_decrypt uses the cover/chapter set to explain
        # what audible-cli actually produced.
        self.assertEqual(artifacts["cover"], Path("/d/Galaxys_Edge_(500).jpg"))

    def test_empty_touched(self):
        self.assertEqual(
            backup.classify_artifacts([]),
            {"audio_file": None, "source_kind": None, "voucher": None, "cover": None},
        )

    def test_cover_extension_jpg_jpeg_png(self):
        # Cover extension is preserved downstream — this asserts the
        # classifier doesn't drop alternative extensions on the way in.
        for ext in (".jpg", ".jpeg", ".png"):
            with self.subTest(ext=ext):
                cover = Path(f"/d/cover{ext}")
                artifacts = backup.classify_artifacts([cover])
                self.assertEqual(artifacts["cover"], cover)


class AaxcKeyIvTest(unittest.TestCase):
    """`aaxc_key_iv` handles both nested-object and JSON-string voucher shapes.

    Fixtures are inline placeholder hex strings — never the bytes from
    a real voucher (those are DRM keys). The function only does JSON
    parsing + dict navigation, so any 32-hex-char string exercises the
    same code path.
    """

    PLACEHOLDER_KEY = "0123456789abcdef0123456789abcdef"
    PLACEHOLDER_IV = "fedcba9876543210fedcba9876543210"

    def _write_voucher(self, payload: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".voucher", delete=False, encoding="utf-8",
        )
        try:
            json.dump(payload, tmp)
            tmp.flush()
        finally:
            tmp.close()
        return Path(tmp.name)

    def test_nested_object_shape(self):
        voucher = self._write_voucher({
            "content_license": {
                "license_response": {
                    "key": self.PLACEHOLDER_KEY,
                    "iv": self.PLACEHOLDER_IV,
                },
            },
        })
        try:
            key, iv = backup.aaxc_key_iv(voucher)
            self.assertEqual(key, self.PLACEHOLDER_KEY)
            self.assertEqual(iv, self.PLACEHOLDER_IV)
        finally:
            voucher.unlink()

    def test_json_string_shape(self):
        inner = json.dumps({"key": self.PLACEHOLDER_KEY, "iv": self.PLACEHOLDER_IV})
        voucher = self._write_voucher({
            "content_license": {"license_response": inner},
        })
        try:
            key, iv = backup.aaxc_key_iv(voucher)
            self.assertEqual(key, self.PLACEHOLDER_KEY)
            self.assertEqual(iv, self.PLACEHOLDER_IV)
        finally:
            voucher.unlink()

    def test_missing_keys_raises(self):
        voucher = self._write_voucher({
            "content_license": {"license_response": {"key": "only-key"}},
        })
        try:
            with self.assertRaises(KeyError):
                backup.aaxc_key_iv(voucher)
        finally:
            voucher.unlink()


class ScrubDecryptSecretsTest(unittest.TestCase):
    """scrub_decrypt_secrets removes audible_key / audible_iv / activation_bytes
    values from text blobs to keep DRM material out of log streams
    (`coding-policy: no-secrets`)."""

    def test_redacts_audible_key(self):
        text = "ffmpeg -audible_key deadbeefcafebabe -i in.aaxc out.m4b"
        scrubbed = backup.scrub_decrypt_secrets(text)
        self.assertNotIn("deadbeefcafebabe", scrubbed)
        self.assertIn("-audible_key <REDACTED>", scrubbed)

    def test_redacts_audible_iv(self):
        text = "args: -audible_iv 0123456789abcdef -i input.aaxc"
        scrubbed = backup.scrub_decrypt_secrets(text)
        self.assertNotIn("0123456789abcdef", scrubbed)
        self.assertIn("-audible_iv <REDACTED>", scrubbed)

    def test_redacts_activation_bytes(self):
        text = "Command failed: ffmpeg -activation_bytes 1a2b3c4d -i x.aax"
        scrubbed = backup.scrub_decrypt_secrets(text)
        self.assertNotIn("1a2b3c4d", scrubbed)
        self.assertIn("-activation_bytes <REDACTED>", scrubbed)

    def test_redacts_multiple_flags_in_one_blob(self):
        text = "-audible_key KEYHEX -audible_iv IVHEX -i in.aaxc"
        scrubbed = backup.scrub_decrypt_secrets(text)
        self.assertNotIn("KEYHEX", scrubbed)
        self.assertNotIn("IVHEX", scrubbed)

    def test_passthrough_when_no_secret_flag(self):
        text = "Invalid data found when processing input"
        self.assertEqual(backup.scrub_decrypt_secrets(text), text)

    def test_empty_input(self):
        self.assertEqual(backup.scrub_decrypt_secrets(""), "")


class ParseActivationBytesTest(unittest.TestCase):
    """parse_activation_bytes turns audible-cli stdout into a hex token, or
    raises an actionable RuntimeError when the input is empty."""

    def test_extracts_single_token(self):
        self.assertEqual(backup.parse_activation_bytes("deadbeef\n"), "deadbeef")

    def test_extracts_trailing_token_when_prefixed(self):
        # Some audible-cli versions / profiles prefix with a log line.
        out = "Activation bytes for profile: deadbeef\n"
        self.assertEqual(backup.parse_activation_bytes(out), "deadbeef")

    def test_empty_stdout_raises_actionable_error(self):
        # Empty stdout would otherwise become an unstructured IndexError
        # from `[-1]` on the empty list and break the IPC JSON contract.
        with self.assertRaises(RuntimeError) as ctx:
            backup.parse_activation_bytes("")
        # Message tells the operator what to do, not just what broke.
        msg = str(ctx.exception)
        self.assertIn("activation-bytes", msg)
        self.assertIn("audible quickstart", msg)

    def test_whitespace_only_raises(self):
        with self.assertRaises(RuntimeError):
            backup.parse_activation_bytes("   \n\t  \n")


class SanitizeFilenameTest(unittest.TestCase):
    """Pin existing sanitize_filename behavior — touched indirectly by the fix."""

    def test_strips_disallowed_chars(self):
        self.assertEqual(
            backup.sanitize_filename("Red Rising: Part 1 (Dramatized!)"),
            "Red Rising Part 1 Dramatized",
        )

    def test_collapses_whitespace(self):
        self.assertEqual(
            backup.sanitize_filename("  Many    Spaces  Here  "),
            "Many Spaces Here",
        )

    def test_caps_at_200_chars(self):
        long_title = "a" * 500
        result = backup.sanitize_filename(long_title)
        self.assertLessEqual(len(result), 200)


if __name__ == "__main__":
    unittest.main()
