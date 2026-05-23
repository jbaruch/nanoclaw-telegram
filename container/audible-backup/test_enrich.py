"""stdlib-unittest tests for enrich.merge_enrichment.

Run with:
  python3 -m unittest test_enrich

The enrich module's `main()` shells out to audible-cli for the bulk
library fetch + per-book chapter calls; those are integration-shaped
and stay out of unit-test scope per the same testing-standards
reasoning as test_backup.py. The merge_enrichment function is the
pure decision logic that runs against each book, and that's where
field-name regressions land (the round-1 review caught
`series_title` vs `series_name` mismatching the schema).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import enrich  # noqa: E402


class MergeEnrichmentTest(unittest.TestCase):
    """merge_enrichment composes PRESERVE_IF_PRESENT + ENRICHABLE_FIELDS rules."""

    def test_enrichable_field_overwrites_existing(self):
        # ENRICHABLE fields are authoritative for fresh API data — a
        # stale local value gets replaced.
        existing = {"rating_average": "3.0", "asin": "B001"}
        fresh = {"rating_average": "4.5", "asin": "B001"}
        merged = enrich.merge_enrichment(existing, fresh)
        self.assertEqual(merged["rating_average"], "4.5")

    def test_preserve_if_present_keeps_non_empty_existing(self):
        # filename / files / user_id / key / purchase_date / chapters
        # are local-managed and stay when already populated.
        existing = {"filename": "Local Path.m4b", "asin": "B001"}
        fresh = {"filename": "API Default Name.m4b", "asin": "B001"}
        merged = enrich.merge_enrichment(existing, fresh)
        self.assertEqual(merged["filename"], "Local Path.m4b")

    def test_preserve_if_present_fills_when_empty(self):
        # Empty string / None / [] / {} count as empty — fresh wins.
        for empty in ("", None, [], {}):
            with self.subTest(empty_value=empty):
                existing = {"filename": empty, "asin": "B001"}
                fresh = {"filename": "From API.m4b", "asin": "B001"}
                merged = enrich.merge_enrichment(existing, fresh)
                self.assertEqual(merged["filename"], "From API.m4b")

    def test_unmanaged_existing_field_is_preserved(self):
        # A field that's neither in PRESERVE_IF_PRESENT nor in
        # ENRICHABLE_FIELDS but already exists locally is left
        # untouched even when fresh has a different value.
        existing = {"custom_local_note": "operator added", "asin": "B001"}
        fresh = {"custom_local_note": "from api", "asin": "B001"}
        merged = enrich.merge_enrichment(existing, fresh)
        self.assertEqual(merged["custom_local_note"], "operator added")

    def test_new_field_from_fresh_is_added(self):
        existing = {"asin": "B001"}
        fresh = {"asin": "B001", "voice_description": "Sharp baritone"}
        merged = enrich.merge_enrichment(existing, fresh)
        self.assertEqual(merged["voice_description"], "Sharp baritone")

    def test_series_fields_now_match_schema(self):
        # Regression guard for the round-1 review fix — the previous
        # ENRICHABLE_FIELDS spelled this as `series_title`, which never
        # matched anything in the backup.py schema; series metadata
        # silently never refreshed on enrichment.
        self.assertIn("series_name", enrich.ENRICHABLE_FIELDS)
        self.assertIn("series_sequence", enrich.ENRICHABLE_FIELDS)
        self.assertIn("series_link", enrich.ENRICHABLE_FIELDS)
        self.assertNotIn("series_title", enrich.ENRICHABLE_FIELDS)

        # And the merge actually picks up a fresh series_name now:
        existing = {"asin": "B001", "series_name": "Old Series"}
        fresh = {"asin": "B001", "series_name": "Stormlight Archive"}
        merged = enrich.merge_enrichment(existing, fresh)
        self.assertEqual(merged["series_name"], "Stormlight Archive")


if __name__ == "__main__":
    unittest.main()
