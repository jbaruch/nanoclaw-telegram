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

import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class FindMissingOnDiskTest(unittest.TestCase):
    """`find_missing_on_disk` surfaces inventory records whose m4b is gone."""

    def _inventory_record(self, asin: str, filename: str, title: str = "") -> dict:
        return {
            "asin": asin,
            "title": title or f"Title {asin}",
            "filename": filename,
            "files": [f"/library/books/{filename}"] if filename else [],
        }

    def test_returns_records_with_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            books_dir = Path(tmp)
            (books_dir / "Present.m4b").write_bytes(b"")
            inventory = [
                self._inventory_record("ASIN1", "Present.m4b", "Present"),
                self._inventory_record("ASIN2", "Gone.m4b", "Gone"),
            ]
            missing = backup.find_missing_on_disk(inventory, books_dir)
            self.assertEqual([r["asin"] for r in missing], ["ASIN2"])

    def test_skips_records_with_empty_filename(self):
        # Records that have never been downloaded carry `filename=""`.
        # They're a distinct state — outside the disk-presence check —
        # otherwise every empty-inventory bootstrap would alert on every
        # row at once.
        with tempfile.TemporaryDirectory() as tmp:
            inventory = [self._inventory_record("ASIN1", "")]
            self.assertEqual(backup.find_missing_on_disk(inventory, Path(tmp)), [])

    def test_skips_records_with_missing_filename_key(self):
        # Defensive: legacy inventory rows imported from older OpenAudible
        # builds may lack the `filename` key entirely. Treated the same
        # as empty-string: skip, not error.
        with tempfile.TemporaryDirectory() as tmp:
            inventory = [{"asin": "ASIN1", "title": "Legacy"}]
            self.assertEqual(backup.find_missing_on_disk(inventory, Path(tmp)), [])

    def test_empty_inventory_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(backup.find_missing_on_disk([], Path(tmp)), [])

    def test_returns_full_record_not_just_asin(self):
        # Operator surface in the JSON output reads `title` and
        # `filename` off the returned record; pin that the full
        # inventory shape passes through so a future refactor toward
        # returning bare ASINs doesn't silently strip the operator's
        # context.
        with tempfile.TemporaryDirectory() as tmp:
            inventory = [self._inventory_record("ASIN1", "Gone.m4b", "Gone Book")]
            missing = backup.find_missing_on_disk(inventory, Path(tmp))
            self.assertEqual(len(missing), 1)
            self.assertEqual(missing[0]["title"], "Gone Book")
            self.assertEqual(missing[0]["filename"], "Gone.m4b")

    def test_does_not_match_directory_at_filename_path(self):
        # A directory at `<books_dir>/<filename>` should NOT be treated
        # as a present m4b — `is_file()` rejects it. Defends against the
        # operator state where they manually created a folder with the
        # same name to organize chapter splits.
        with tempfile.TemporaryDirectory() as tmp:
            books_dir = Path(tmp)
            (books_dir / "Confusing.m4b").mkdir()
            inventory = [self._inventory_record("ASIN1", "Confusing.m4b")]
            missing = backup.find_missing_on_disk(inventory, books_dir)
            self.assertEqual([r["asin"] for r in missing], ["ASIN1"])


class LoadSkiplistTest(unittest.TestCase):
    """`load_skiplist` parses the operator's ASIN allowlist file."""

    def _write_skiplist(self, content: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        )
        try:
            tmp.write(content)
            tmp.flush()
        finally:
            tmp.close()
        return Path(tmp.name)

    def test_parses_one_asin_per_line(self):
        path = self._write_skiplist("ASIN1\nASIN2\nASIN3\n")
        try:
            self.assertEqual(
                backup.load_skiplist(path), {"ASIN1", "ASIN2", "ASIN3"},
            )
        finally:
            path.unlink()

    def test_strips_inline_hash_comments(self):
        # Operator UX target from the issue: `# Title hint` next to the
        # bare ASIN so a future glance at the file remembers why the
        # ASIN was filtered without needing to cross-reference inventory.
        path = self._write_skiplist(
            "1663704015  # Healing the Wounds (Plus-only)\n"
            "B079LRSMNN  # Galaxy's Edge\n"
        )
        try:
            self.assertEqual(
                backup.load_skiplist(path), {"1663704015", "B079LRSMNN"},
            )
        finally:
            path.unlink()

    def test_skips_blank_lines(self):
        path = self._write_skiplist("\nASIN1\n\n\nASIN2\n")
        try:
            self.assertEqual(backup.load_skiplist(path), {"ASIN1", "ASIN2"})
        finally:
            path.unlink()

    def test_skips_pure_comment_lines(self):
        path = self._write_skiplist("# header comment\nASIN1\n# trailing\n")
        try:
            self.assertEqual(backup.load_skiplist(path), {"ASIN1"})
        finally:
            path.unlink()

    def test_missing_file_returns_empty_set(self):
        # Bootstrap state, not an error — every weekly run before the
        # operator's first explicit-ack must work without surfacing a
        # filesystem error.
        self.assertEqual(
            backup.load_skiplist(Path("/nonexistent/skiplist.txt")),
            set(),
        )


class FindNewBooksSkiplistTest(unittest.TestCase):
    """`find_new_books` honours the skiplist set."""

    def test_filters_out_skiplist_asins(self):
        library = [
            {"asin": "ASIN1", "title": "New"},
            {"asin": "ASIN2", "title": "Skipped"},
        ]
        new = backup.find_new_books(
            library, known_asins=set(), skiplist_asins={"ASIN2"},
        )
        self.assertEqual([b["asin"] for b in new], ["ASIN1"])

    def test_no_skiplist_arg_preserves_legacy_behaviour(self):
        # Callers from before the skiplist landed pass two positional
        # args; pin that the third arg defaults to "no filter applied"
        # so we don't regress the #631 fix's call site.
        library = [{"asin": "ASIN1", "title": "New"}]
        new = backup.find_new_books(library, known_asins=set())
        self.assertEqual([b["asin"] for b in new], ["ASIN1"])

    def test_skiplist_and_known_both_filter(self):
        library = [
            {"asin": "ASIN1", "title": "Already have"},
            {"asin": "ASIN2", "title": "Skipped"},
            {"asin": "ASIN3", "title": "New"},
        ]
        new = backup.find_new_books(
            library, known_asins={"ASIN1"}, skiplist_asins={"ASIN2"},
        )
        self.assertEqual([b["asin"] for b in new], ["ASIN3"])


class MainEmptyLibraryJsonTest(unittest.TestCase):
    """Smoke test for `main()` on the no-new-books JSON path.

    Catches the regression a prior revision of this PR shipped: the
    `skipped_by_filter` audit-list was being computed against `library`
    *before* `library = fetch_enriched_library(...)` ran, so any normal
    invocation raised `NameError` before producing output. The helper
    tests don't exercise `main()`, so without this case the bug would
    pass CI again on the next refactor.
    """

    def _run_main(self, library_dir: Path) -> dict:
        # Stub fetch_enriched_library so the test stays offline (no
        # audible-cli, no network). Returns an empty library — covers the
        # no-new-books JSON branch where the regression originally fired.
        stdout = io.StringIO()
        with patch.object(backup, "fetch_enriched_library", return_value=[]), \
             patch.object(sys, "argv", ["backup.py", "--json"]), \
             patch.dict(os.environ, {"AUDIOBOOK_DIR": str(library_dir)}), \
             patch.object(sys, "stdout", stdout):
            backup.main()
        return json.loads(stdout.getvalue())

    def test_no_new_books_emits_full_json_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            library_dir = Path(tmp)
            (library_dir / "books.json").write_text("[]")
            payload = self._run_main(library_dir)
            self.assertEqual(payload["new_books"], 0)
            self.assertEqual(payload["downloaded"], 0)
            self.assertEqual(payload["skipped"], 0)
            self.assertEqual(payload["failed"], 0)
            self.assertEqual(payload["missing_on_disk"], 0)
            self.assertEqual(payload["skipped_by_filter"], [])
            self.assertEqual(payload["books"], [])

    def test_skiplist_present_surfaces_filter_audit(self):
        # Inventory is empty (books.json = []), so the skiplist is the
        # sole filter keeping the synthesized library ASINs out of the
        # new-books bucket. Proves the audit field reflects the filter
        # even when nothing makes it through the download branch.
        with tempfile.TemporaryDirectory() as tmp:
            library_dir = Path(tmp)
            (library_dir / "books.json").write_text("[]")
            (library_dir / backup.SKIPLIST_FILENAME).write_text(
                "ASIN_SKIP_1\nASIN_SKIP_2\n", encoding="utf-8",
            )
            library_fixture = [
                {"asin": "ASIN_SKIP_1", "title": "Skip me 1"},
                {"asin": "ASIN_SKIP_2", "title": "Skip me 2"},
            ]
            stdout = io.StringIO()
            with patch.object(
                backup, "fetch_enriched_library", return_value=library_fixture,
            ), patch.object(sys, "argv", ["backup.py", "--json"]), \
               patch.dict(os.environ, {"AUDIOBOOK_DIR": str(library_dir)}), \
               patch.object(sys, "stdout", stdout):
                backup.main()
            payload = json.loads(stdout.getvalue())
            self.assertEqual(
                payload["skipped_by_filter"], ["ASIN_SKIP_1", "ASIN_SKIP_2"],
            )
            self.assertEqual(payload["books"], [])


if __name__ == "__main__":
    unittest.main()
