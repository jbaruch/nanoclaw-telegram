#!/usr/bin/env python3
"""Audible audiobook backup.

Maintains books (1).json as the canonical library inventory, using audible-cli
as the source of truth for "what's in my Audible account." Downloads new
purchases, decrypts to M4B, and appends enriched records to the inventory.

Usage:
    python3 backup.py [--dry-run] [--json]

Mounts expected:
    /library          — audiobook library root
    /root/.audible    — audible-cli auth dir (the orchestrator mounts the
                        host's ~/.audible here so the cli's default
                        home-dir lookup finds the credentials)

Environment:
    AUDIOBOOK_DIR — override library path (default: /library)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# Audible API response groups for enriched library data.
# listening_status is REQUIRED — it's the source of is_finished and
# percent_complete which we need for read_status mapping.
LIBRARY_RESPONSE_GROUPS = (
    "contributors,product_desc,product_extended_attrs,product_attrs,"
    "media,rating,series,categories,category_ladders,listening_status,"
    "product_details,reviews,relationships"
)


# Required fields the API must return on every book. If any are missing,
# the script fails loudly so we know the API contract changed.
REQUIRED_API_FIELDS = ("asin", "title")

# Fields we expect to populate in every output record. Used by validation
# to catch silent data loss.
REQUIRED_OUTPUT_FIELDS = (
    "asin", "product_id", "title", "author", "narrated_by", "duration",
    "seconds", "release_date", "image_url", "info_link", "author_link",
    "abridged", "ayce", "read_status", "genre", "key", "filename", "files",
    "rating_average", "rating_count", "summary", "description", "copyright",
    "language", "region", "publisher", "purchase_date", "title_short",
    "user_id", "download_link",
    # Series + extras
    "series_name", "series_sequence", "series_link", "subtitle",
    "percent_complete", "isbn", "sku", "voice_description", "narration_accent",
    "format_type", "content_type", "publication_datetime", "date_added",
    "performance_rating", "performance_rating_count",
    "story_rating", "story_rating_count",
    "chapters",
)


class FieldValidationError(Exception):
    """Raised when a record's fields don't match the expected schema."""
    pass


def py_bool_str(value) -> str:
    """OpenAudible-compatible bool string: 'true' / 'false' (lowercase)."""
    return "true" if value else "false"


def safe_str(value, default: str = "") -> str:
    """Return value as string, or default if None/empty."""
    if value is None or value == "":
        return default
    return str(value)


def validate_record(record: dict, asin: str) -> None:
    """Raise FieldValidationError if record is missing required fields.

    Empty values are OK (some books don't have all metadata), but the
    KEY must be present so downstream consumers can rely on the schema.
    """
    missing = [f for f in REQUIRED_OUTPUT_FIELDS if f not in record]
    if missing:
        raise FieldValidationError(
            f"Record for {asin} missing required fields: {missing}"
        )


def find_inventory(library_dir: Path) -> Optional[Path]:
    """Find the inventory JSON. Google Drive may rename to 'books (1).json'."""
    for name in ["books (1).json", "books.json"]:
        p = library_dir / name
        if p.exists():
            return p
    return None


def load_inventory(inventory_path: Path) -> list[dict]:
    """Load existing inventory."""
    return json.loads(inventory_path.read_text())


def save_inventory(inventory_path: Path, books: list[dict]) -> None:
    """Save inventory atomically (write to temp + rename)."""
    tmp = inventory_path.with_suffix(inventory_path.suffix + ".tmp")
    tmp.write_text(json.dumps(books, indent=2, ensure_ascii=False))
    tmp.replace(inventory_path)


def fetch_enriched_library(tmp_dir: Path) -> list[dict]:
    """Fetch full Audible library with enriched metadata via API.

    Uses /1.0/library with response_groups for full field set, paginating
    through results. Returns all owned items.
    """
    all_items: list[dict] = []
    page = 1
    page_size = 1000
    while True:
        out = tmp_dir / f"library-page-{page}.json"
        subprocess.run(
            [
                "audible", "api", "/1.0/library",
                "-p", f"response_groups={LIBRARY_RESPONSE_GROUPS}",
                "-p", f"num_results={page_size}",
                "-p", f"page={page}",
                "-o", str(out),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(out.read_text())
        items = data.get("items", [])
        if not items:
            break
        all_items.extend(items)
        if len(items) < page_size:
            break
        page += 1
    return all_items


def fetch_chapters(asin: str) -> list[dict]:
    """Fetch chapter list for a book via /1.0/content/{asin}/metadata.

    Returns OpenAudible-compatible chapter list. Returns empty list on
    failure WITH a stderr-logged diagnostic naming the ASIN + error type
    — chapter metadata is non-fatal (the M4B + cover still complete and
    OpenAudible can refill chapters later via enrich), but a silent
    empty-list return would leave a future operator unable to tell
    "this title has no chapters" from "the chapter API broke".
    """
    try:
        result = subprocess.run(
            [
                "audible", "api", f"/1.0/content/{asin}/metadata",
                "-p", "response_groups=chapter_info",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout)
        return data.get("content_metadata", {}).get("chapter_info", {}).get("chapters", [])
    except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
        log(f"WARN: chapter fetch failed for {asin} ({type(e).__name__}: {e}); proceeding with empty chapters")
        return []


def get_known_asins(books: list[dict]) -> set[str]:
    """Extract ASINs from inventory."""
    return {b["asin"] for b in books if b.get("asin")}


def find_new_books(library: list[dict], known_asins: set[str]) -> list[dict]:
    """Find books in Audible library not yet in inventory."""
    return [
        book for book in library
        if book.get("asin") and book["asin"] not in known_asins
    ]


def find_missing_on_disk(inventory: list[dict], books_dir: Path) -> list[dict]:
    """Find inventory records whose m4b file is no longer present under books_dir.

    Soft-alert dedup gap: `find_new_books` keys off ASIN-presence in
    inventory, so once a record exists no later run re-downloads even if
    the m4b is removed (Google Drive sync hiccup, manual cleanup, partial
    restore, disk migration). This helper surfaces inventory rows with a
    non-empty `filename` whose file is missing under `books_dir`, so the
    operator sees the gap on the next weekly run instead of finding out
    at playback time.

    Records with an empty `filename` are skipped — those have never been
    downloaded, a distinct state outside the disk-presence check.
    """
    missing: list[dict] = []
    for record in inventory:
        filename = record.get("filename") or ""
        if not filename:
            continue
        if not (books_dir / filename).is_file():
            missing.append(record)
    return missing


def map_to_inventory_schema(api_book: dict, region: str = "US") -> dict:
    """Map enriched audible-cli API response to inventory JSON schema.

    Field conventions match OpenAudible's books.json schema:
    - Bools serialized as 'true'/'false' lowercase strings
    - None/missing values become empty strings (not 'None')
    - URLs use audible.com format

    Raises FieldValidationError if required API fields are missing.
    """
    asin = api_book.get("asin", "")
    title = safe_str(api_book.get("title"))

    # Validate required API fields
    for field in REQUIRED_API_FIELDS:
        if not api_book.get(field):
            raise FieldValidationError(
                f"API response missing required field '{field}' "
                f"for ASIN={asin or '(no asin)'}"
            )

    # Authors and narrators come as lists of {asin, name} dicts
    authors = api_book.get("authors") or []
    narrators = api_book.get("narrators") or []
    author_str = ", ".join(a.get("name", "") for a in authors if a.get("name"))
    narrator_str = ", ".join(n.get("name", "") for n in narrators if n.get("name"))

    # Categories: each "ladder" is a hierarchy path (e.g.,
    # ["Mystery, Thriller & Suspense", "Crime Fiction"]).
    # OpenAudible format: "Top:Sub:SubSub" per path, multiple paths comma-separated.
    genre_paths: list[str] = []
    for ladder in api_book.get("category_ladders") or []:
        names = [
            cat.get("name") for cat in (ladder.get("ladder") or [])
            if cat.get("name")
        ]
        if names:
            path = ":".join(names)
            if path not in genre_paths:
                genre_paths.append(path)
    genre_str = ", ".join(genre_paths)

    # Rating is nested. Three separate scores: overall, performance (narration),
    # story. Useful signal for recommendations (a book may rate high overall
    # but be carried entirely by performance, or vice versa).
    rating = api_book.get("rating") or {}
    overall = rating.get("overall_distribution") or {}
    perf = rating.get("performance_distribution") or {}
    story = rating.get("story_distribution") or {}
    rating_average = overall.get("average_rating", "")
    rating_count = overall.get("num_ratings", "") or rating.get("num_reviews", "")
    perf_rating = perf.get("average_rating", "")
    perf_count = perf.get("num_ratings", "")
    story_rating = story.get("average_rating", "")
    story_count = story.get("num_ratings", "")

    # Duration: API gives runtime_length_min
    runtime_min = api_book.get("runtime_length_min") or 0
    runtime_sec = int(runtime_min) * 60 if runtime_min else 0
    duration_str = (
        f"{int(runtime_min) // 60:02d}:{int(runtime_min) % 60:02d}:00"
        if runtime_min else ""
    )

    # Series — first series in list (book may be in multiple)
    series_list = api_book.get("series") or []
    series_first = series_list[0] if series_list else {}
    series_name = safe_str(series_first.get("title"))
    series_sequence = safe_str(series_first.get("sequence"))
    series_asin = safe_str(series_first.get("asin"))
    series_link = f"/series/{series_name.replace(' ', '-')}/{series_asin}" if series_asin else ""

    # Author link — OpenAudible uses audible.com/author/{Name}/{ASIN}
    author_link = ""
    if authors and authors[0].get("asin") and authors[0].get("name"):
        name_url = authors[0]["name"].replace(" ", "+")
        author_link = f"https://www.audible.com/author/{name_url}/{authors[0]['asin']}"

    # info_link — construct from ASIN; API may return None
    api_info_link = api_book.get("product_page_url")
    info_link = api_info_link if api_info_link else f"https://www.audible.com/pd/{asin}"

    # listening_status nested object — required for read_status mapping
    listening = api_book.get("listening_status") or {}
    is_finished = listening.get("is_finished") or api_book.get("is_finished")
    pct = listening.get("percent_complete") or api_book.get("percent_complete") or 0

    # read_status: OpenAudible 3-state convention (Finished/Reading/Unread)
    if is_finished:
        read_status = "Finished"
    elif pct and float(pct) > 0:
        read_status = "Reading"
    else:
        read_status = "Unread"

    record = {
        # --- Identity ---
        "asin": asin,
        "product_id": safe_str(api_book.get("sku") or asin),
        "key": safe_str(api_book.get("sku") or asin),

        # --- Titles ---
        "title": title,
        "title_short": title[:200],
        "subtitle": safe_str(api_book.get("subtitle")),

        # --- Contributors ---
        "author": author_str,
        "author_link": author_link,
        "narrated_by": narrator_str,
        "voice_description": safe_str(api_book.get("voice_description")),
        "narration_accent": safe_str(api_book.get("narration_accent")),

        # --- Publisher / metadata ---
        "publisher": safe_str(api_book.get("publisher_name")),
        "summary": safe_str(api_book.get("merchandising_summary")),
        "description": safe_str(api_book.get("publisher_summary")),
        "copyright": safe_str(api_book.get("copyright")),
        "language": safe_str(api_book.get("language")),
        "region": region,
        "genre": genre_str,
        "format_type": safe_str(api_book.get("format_type")),
        "content_type": safe_str(api_book.get("content_type")),
        "isbn": safe_str(api_book.get("isbn")),
        "sku": safe_str(api_book.get("sku")),

        # --- Ratings (overall + per-aspect for recommendation signal) ---
        "rating_average": str(rating_average) if rating_average else "",
        "rating_count": str(rating_count) if rating_count else "",
        "performance_rating": str(perf_rating) if perf_rating else "",
        "performance_rating_count": str(perf_count) if perf_count else "",
        "story_rating": str(story_rating) if story_rating else "",
        "story_rating_count": str(story_count) if story_count else "",

        # --- Dates ---
        "purchase_date": safe_str(api_book.get("purchase_date")),
        "release_date": safe_str(
            api_book.get("release_date") or api_book.get("issue_date")
        ),
        "publication_datetime": safe_str(api_book.get("publication_datetime")),
        # date_added is nested under library_status, not top-level
        "date_added": safe_str(
            (api_book.get("library_status") or {}).get("date_added")
            or api_book.get("date_added")
        ),

        # --- Duration ---
        "duration": duration_str,
        "seconds": runtime_sec,

        # --- Format flags (OpenAudible: lowercase string bools) ---
        "abridged": py_bool_str(api_book.get("format_type") != "unabridged"),
        "ayce": py_bool_str(api_book.get("is_ayce")),

        # --- Read status ---
        "read_status": read_status,
        "percent_complete": float(pct) if pct else 0,

        # --- Series (OpenAudible convention) ---
        "series_name": series_name,
        "series_sequence": series_sequence,
        "series_link": series_link,

        # --- Media URLs ---
        "image_url": safe_str(
            (api_book.get("product_images") or {}).get("500")
            or api_book.get("cover_url")
        ),
        "info_link": info_link,

        # --- User-specific (preserved by enrich) ---
        "user_id": "",
        "download_link": "",

        # --- Local file paths (populated after download) ---
        "filename": "",
        "files": [],

        # --- Chapters (populated separately via fetch_chapters) ---
        "chapters": [],
    }

    validate_record(record, asin)
    return record


def log(msg: str) -> None:
    """Print progress to stderr so stdout stays clean for --json output."""
    print(msg, file=sys.stderr)


def sanitize_filename(title: str) -> str:
    """Make a filesystem-safe filename from a book title."""
    safe = "".join(c for c in title if c.isalnum() or c in " ._-")
    return " ".join(safe.split())[:200]


_activation_bytes_cache: Optional[str] = None


def parse_activation_bytes(stdout: str) -> str:
    """Extract the activation-bytes hex token from `audible activation-bytes` stdout.

    Pure helper around `audible-cli`'s output shape — splits on
    whitespace and returns the trailing token. Raises RuntimeError with
    an actionable message when stdout is empty / contains no token, so
    the JSON process contract this script holds with the IPC handler
    surfaces an operator-readable failure instead of an unstructured
    `IndexError` from `[-1]` on an empty list
    (`coding-policy: error-handling` Actionable Messages).
    """
    tokens = stdout.strip().split()
    if not tokens:
        raise RuntimeError(
            "audible-cli `activation-bytes` returned empty stdout — "
            "verify the auth profile is current via "
            "`audible -P <profile> activation-bytes` from inside the "
            "container, and re-run `audible quickstart` if the token "
            "needs refreshing"
        )
    return tokens[-1]


def get_activation_bytes() -> str:
    """Fetch (and cache) AAX activation bytes from the audible-cli profile.

    audible-cli ≥0.3 removed the `decrypt` subcommand, so the script now
    invokes ffmpeg directly with the per-account activation bytes for
    AAX content. The bytes are stable per Audible account, so we fetch
    them once per backup run and reuse across books.
    """
    global _activation_bytes_cache
    if _activation_bytes_cache is not None:
        return _activation_bytes_cache
    result = subprocess.run(
        ["audible", "activation-bytes"],
        capture_output=True, text=True, check=True,
    )
    _activation_bytes_cache = parse_activation_bytes(result.stdout)
    return _activation_bytes_cache


def aaxc_key_iv(voucher_path: Path) -> tuple[str, str]:
    """Extract decryption key + IV from an audible-cli AAXC voucher.

    audible-cli stores the decrypted license response in
    `content_license.license_response` as either a string of JSON
    (older shapes) or a nested object containing `key` and `iv` hex
    strings (current shape). Try the nested form first and fall back
    to JSON-parsing a string payload.
    """
    payload = json.loads(voucher_path.read_text())
    license_resp = (
        payload.get("content_license", {}).get("license_response") or {}
    )
    if isinstance(license_resp, str):
        license_resp = json.loads(license_resp)
    return license_resp["key"], license_resp["iv"]


def parse_skipped_paths(combined_output: str) -> list[Path]:
    """Extract file paths from audible-cli's `Skip download` log lines.

    Recovery-path helper: when audible-cli sees the target file already
    on disk it prints `File <path> already exists. Skip download.` and
    bumps no mtimes. Without this path the mtime-based discovery in
    download_and_decrypt would mis-fail a recovery run as
    "no audio file" even when every file is present.
    """
    return [
        Path(p) for p in re.findall(r"File (\S+) already exists", combined_output)
    ]


def classify_artifacts(touched: list[Path]) -> dict:
    """Bucket audible-cli's downloaded files by role.

    Returns a dict with `audio_file` (preferred AAXC over AAX), the
    detected `source_kind` (`aax` | `aaxc` | `None`), `voucher`, and
    `cover` paths. AAXC ranks higher because it's encrypted with a
    per-title voucher that ships in the download, while AAX needs the
    per-account activation bytes; when audible-cli happens to provide
    both for the same title the AAXC path is more robust. Returns an
    empty audio_file when no `.aax`/`.aaxc` is in the set.
    """
    audio_files = [f for f in touched if f.suffix in (".aax", ".aaxc")]
    audio_files.sort(key=lambda f: (0 if f.suffix == ".aaxc" else 1, f.name))
    audio_file = audio_files[0] if audio_files else None
    source_kind = audio_file.suffix.lstrip(".") if audio_file else None
    voucher = next((f for f in touched if f.suffix == ".voucher"), None)
    cover = next(
        (f for f in touched if f.suffix.lower() in (".jpg", ".jpeg", ".png")),
        None,
    )
    return {
        "audio_file": audio_file,
        "source_kind": source_kind,
        "voucher": voucher,
        "cover": cover,
    }


_DECRYPT_SECRET_FLAGS = ("-audible_key", "-audible_iv", "-activation_bytes")


def scrub_decrypt_secrets(text: str) -> str:
    """Redact AAXC key / IV / activation-bytes values from a text blob.

    ffmpeg receives those tokens as `<flag> <hex>` argv pairs. If the
    blob contains one of the flag literals followed by a token, the
    token gets replaced with `<REDACTED>`. Defensive belt-and-suspenders
    for log lines that might surface the ffmpeg command tail — per
    `coding-policy: no-secrets`, the per-account activation bytes and
    per-file voucher key/IV never reach any log stream.
    """
    out = text
    for flag in _DECRYPT_SECRET_FLAGS:
        out = re.sub(
            rf"({re.escape(flag)})(\s+)\S+",
            r"\1\2<REDACTED>",
            out,
        )
    return out


def ffmpeg_decrypt(
    audio_file: Path, source_kind: str, voucher_file: Optional[Path],
    output_m4b: Path,
) -> subprocess.CompletedProcess:
    """Decrypt an AAX/AAXC file to M4B via ffmpeg.

    AAX: needs per-account activation bytes (-activation_bytes <hex>).
    AAXC: needs per-file key + IV from the .voucher JSON (-audible_key
          / -audible_iv). The `-c copy` flag preserves the AAC stream
          and chapter metadata without re-encoding.
    """
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if source_kind == "aax":
        cmd += ["-activation_bytes", get_activation_bytes()]
    elif source_kind == "aaxc":
        if voucher_file is None:
            raise ValueError("AAXC source requires a voucher file")
        key, iv = aaxc_key_iv(voucher_file)
        cmd += ["-audible_key", key, "-audible_iv", iv]
    else:
        raise ValueError(f"unsupported source kind: {source_kind}")
    cmd += ["-i", str(audio_file), "-c", "copy", str(output_m4b)]
    return subprocess.run(cmd, capture_output=True, text=True)


def download_and_decrypt(
    book: dict, download_dir: Path, books_dir: Path,
    aax_dir: Path, art_dir: Path,
) -> dict:
    """Download and decrypt a single book.

    Returns book dict augmented with status, m4b_path, filename, files.
    """
    asin = book["asin"]
    title = book.get("title", "Unknown")
    log(f"\n--- Downloading: {title} ({asin}) ---")

    # audible-cli's output naming has drifted across releases — some
    # builds prefix with the ASIN, some with the title plus a bitrate
    # suffix (e.g. `Red_Rising-LC_64_22050_stereo.aax`). The legacy
    # `{asin}*` glob missed every title-prefix variant, so every weekly
    # run for ~8 weeks redownloaded the same ~5 GB and reported "no
    # audio file found" for ASINs whose files were in fact on disk.
    # Capture a pre-download timestamp and discover artifacts by mtime
    # instead of by filename prefix — same approach scripts/audible-backup.sh
    # took in commit 8d07215d.
    before_download = time.time() - 1  # -1s margin for FS timestamp resolution

    # Download
    result = subprocess.run(
        [
            "audible", "download",
            "--asin", asin,
            "--output-dir", str(download_dir),
            "--aax-fallback",
            "--cover", "--cover-size", "500", "--chapter",
            "--no-confirm",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log(f"FAILED to download {asin}: {result.stderr[-500:]}")
        return {**book, "status": "failed", "error": "download failed"}

    if "not downloadable" in result.stdout + result.stderr:
        log(f"SKIPPED: {title} — not downloadable (not owned or Plus-only)")
        return {**book, "status": "skipped", "error": "not downloadable"}

    # Collect everything audible-cli created during this download.
    touched = sorted(
        f for f in download_dir.iterdir()
        if f.is_file() and f.stat().st_mtime > before_download
    )

    # Recovery / idempotent-retry case: audible-cli skips download when
    # the target file already exists, prints `File <path> already
    # exists. Skip download.` for each, and bumps no mtimes. The
    # mtime-based discovery above will return empty, but the files are
    # present and decryptable. Parse stdout for the existing-file
    # paths so a re-run on top of a previous partial state proceeds
    # instead of mis-failing as "no audio file".
    if not touched:
        touched = sorted(
            p for p in parse_skipped_paths(result.stdout + result.stderr)
            if p.is_file()
        )

    artifacts = classify_artifacts(touched)
    audio_file = artifacts["audio_file"]
    source_kind = artifacts["source_kind"]
    voucher = artifacts["voucher"]
    cover_file = artifacts["cover"]

    if audio_file is None:
        log(f"FAILED: no audio file found for {asin}")
        if touched:
            log("  files touched in this run:")
            for f in touched:
                log(f"    {f}")
        return {
            **book,
            "status": "failed",
            "error": "no audio file after download",
            "touched_files": [str(f) for f in touched],
        }

    if source_kind == "aaxc" and voucher is None:
        log(f"FAILED: AAXC source for {asin} requires a .voucher but none was downloaded")
        return {
            **book,
            "status": "failed",
            "error": "AAXC source missing .voucher",
            "source_file": str(audio_file),
        }

    safe_title = sanitize_filename(title)
    output_m4b = books_dir / f"{safe_title}.m4b"

    # Decrypt via ffmpeg. audible-cli ≥0.3 removed the `decrypt`
    # subcommand the script used to call; ffmpeg has had Audible AAX
    # support since 2.8 and AAXC support since 4.4, both of which are
    # present in the slim image's ffmpeg.
    log(f"Decrypting {source_kind.upper()} to M4B via ffmpeg...")
    try:
        result = ffmpeg_decrypt(audio_file, source_kind, voucher, output_m4b)
    except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as e:
        log(f"FAILED to prepare decrypt for {asin}: {e}")
        return {**book, "status": "failed", "error": f"decrypt prep failed: {e}"}
    if result.returncode != 0 or not output_m4b.exists():
        # Scrub the decryption-material tokens (`-audible_key <hex>`,
        # `-audible_iv <hex>`, `-activation_bytes <hex>`) out of the
        # ffmpeg stderr before logging, per
        # `coding-policy: no-secrets` ("Never log secrets — not at any
        # log level, not in error messages, not in stack traces").
        # ffmpeg's own error output does not normally echo the command
        # line, but its argv-rejection paths can, and a future ffmpeg
        # version could change that behavior; defense-in-depth keeps
        # the AAXC key, IV, and per-account activation bytes out of
        # log streams regardless.
        scrubbed = scrub_decrypt_secrets(result.stderr)[-500:]
        log(f"FAILED to decrypt {asin}: {scrubbed}")
        return {**book, "status": "failed", "error": "decrypt failed"}

    log(f"OK: {safe_title}.m4b")

    # Cover art — preserve the source extension. The shell-script
    # equivalent does the same; the prior Python code force-renamed
    # every cover to `.jpg`, which left consumers that sniff by
    # extension confused when audible-cli delivered a `.png` or
    # `.jpeg`.
    cover_path = ""
    if cover_file is not None and cover_file.is_file():
        cover_ext = cover_file.suffix.lower()
        cover_dest = art_dir / f"{safe_title}{cover_ext}"
        cover_dest.write_bytes(cover_file.read_bytes())
        cover_path = str(cover_dest)

    # Archive raw AAX/AAXC + voucher. The voucher is required to
    # re-decrypt the AAXC, so it must be co-located with the archived
    # source; without it the archive is effectively a brick. (The shell
    # script's archive path already did this; the Python sidecar never
    # did, so every prior AAXC backup left an un-re-decryptable
    # archive.)
    aax_dest = aax_dir / audio_file.name
    audio_file.rename(aax_dest)
    voucher_archived: Optional[Path] = None
    if source_kind == "aaxc" and voucher is not None and voucher.is_file():
        voucher_archived = aax_dir / voucher.name
        voucher.rename(voucher_archived)

    # Cleanup this book's leftover tmp_download files. Use the
    # `touched` list directly rather than re-walking by mtime: in the
    # recovery path (audible-cli skipped a redundant download) the
    # discovered files have old mtimes and an mtime-only sweep would
    # leave them stranded across runs. The audio file is already moved
    # out via rename above, so it's safely filtered by `is_file()`.
    for f in touched:
        if f.is_file() and f.parent == download_dir:
            try:
                f.unlink()
            except OSError as e:
                log(f"WARN: cleanup failed for {f}: {e}")

    # Populate post-download fields
    files_list = [str(output_m4b)]
    if cover_path:
        files_list.append(cover_path)
    files_list.append(str(aax_dest))

    return {
        **book,
        "status": "ok",
        "m4b_path": str(output_m4b),
        "filename": output_m4b.name,
        "files": files_list,
    }


def main():
    parser = argparse.ArgumentParser(description="Audible audiobook backup")
    parser.add_argument("--dry-run", action="store_true",
                        help="List new books without downloading")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()

    library_dir = Path(os.environ.get("AUDIOBOOK_DIR", "/library"))
    tmp_dir = Path("/tmp/audible-backup")
    tmp_dir.mkdir(exist_ok=True)

    inventory_path = find_inventory(library_dir)
    if not inventory_path:
        msg = "Inventory file not found (looked for books (1).json, books.json)"
        print(json.dumps({"error": msg}) if args.json else f"ERROR: {msg}")
        sys.exit(1)

    books_dir = library_dir / "books"
    aax_dir = library_dir / "aax"
    art_dir = library_dir / "art"
    download_dir = library_dir / "tmp_download"

    # Load existing inventory
    inventory = load_inventory(inventory_path)
    known = get_known_asins(inventory)

    if not args.json:
        print("=== Audible Backup ===")
        print(f"Inventory: {inventory_path.name} ({len(known)} books)")
        print("Fetching Audible library (enriched metadata)...")

    # Fetch enriched library
    library = fetch_enriched_library(tmp_dir)
    new_books_raw = find_new_books(library, known)

    # Soft-alert disk-presence check: inventory records whose m4b is
    # missing on disk get surfaced as `status: "missing_on_disk"` so the
    # operator sees the gap before reaching for playback. No automatic
    # redownload — preserves operator-enriched metadata and lets a slow
    # Google Drive sync rehydrate the file on next access.
    missing_records = find_missing_on_disk(inventory, books_dir)
    missing_results = [
        {
            "asin": r.get("asin", ""),
            "title": r.get("title", ""),
            "filename": r.get("filename", ""),
            "status": "missing_on_disk",
        }
        for r in missing_records
    ]

    if not args.json:
        print(f"Audible library: {len(library)}")
        print(f"New books to download: {len(new_books_raw)}")
        for b in new_books_raw:
            authors = ", ".join(
                a.get("name", "") for a in (b.get("authors") or [])
            )
            print(f"  {b.get('asin')}  {b.get('title', '?')} — {authors}")
        if missing_records:
            print(f"Missing on disk: {len(missing_records)}")
            for r in missing_records:
                print(f"  {r.get('asin')}  {r.get('title', '?')} — {r.get('filename', '?')}")

    if not new_books_raw:
        if args.json:
            print(json.dumps({
                "new_books": 0,
                "downloaded": 0,
                "failed": 0,
                "missing_on_disk": len(missing_results),
                "books": missing_results,
            }))
        else:
            print("Library up to date.")
        return

    # Map to inventory schema
    new_records = [map_to_inventory_schema(b) for b in new_books_raw]

    if args.dry_run:
        if args.json:
            print(json.dumps({
                "new_books": len(new_records),
                "dry_run": True,
                "missing_on_disk": len(missing_results),
                "books": new_records + missing_results,
            }))
        else:
            print("\nDRY RUN — nothing downloaded.")
        return

    # Download and decrypt
    books_dir.mkdir(exist_ok=True)
    aax_dir.mkdir(exist_ok=True)
    art_dir.mkdir(exist_ok=True)
    download_dir.mkdir(exist_ok=True)

    downloaded = 0
    failed = 0
    skipped = 0
    results = []

    for record in new_records:
        # Fetch chapters before download (cheap API call)
        record["chapters"] = fetch_chapters(record["asin"])

        result = download_and_decrypt(
            record, download_dir, books_dir, aax_dir, art_dir
        )
        results.append(result)

        if result["status"] == "ok":
            downloaded += 1
            # Append to inventory and persist immediately so a crash mid-batch
            # doesn't lose successful downloads
            inventory.append({
                k: v for k, v in result.items()
                if k not in ("status", "error", "m4b_path")
            })
            save_inventory(inventory_path, inventory)
        elif result["status"] == "skipped":
            skipped += 1
        else:
            failed += 1

    # Cleanup empty download dir
    try:
        download_dir.rmdir()
    except OSError:
        pass

    if args.json:
        print(json.dumps({
            "new_books": len(new_records),
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
            "missing_on_disk": len(missing_results),
            "books": results + missing_results,
        }))
    else:
        print(f"\n=== Backup complete ===")
        print(f"Downloaded: {downloaded}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {failed}")
        print(f"Missing on disk: {len(missing_results)}")
        print(f"Inventory now has: {len(inventory)} books")

    # Non-zero exit on any download/decrypt failure so the cadence
    # wrapper records this as failed instead of silent success — the
    # silent-loop that hid the title-prefix bug for ~8 weeks of stale
    # books-library.csv. "skipped" books (not downloadable / Plus-only)
    # are expected and do NOT trigger non-zero exit.
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
