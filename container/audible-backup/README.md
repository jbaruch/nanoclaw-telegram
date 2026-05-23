# audible-backup

Dockerized Audible audiobook backup. Downloads new purchases, decrypts to M4B with chapters. Migrates from an existing OpenAudible library without re-downloading.

## What it does

1. Reads your existing OpenAudible `books.json` for known ASINs
2. Fetches your current Audible library via [audible-cli](https://github.com/mkb79/audible-cli)
3. Finds new purchases not in the inventory
4. Downloads AAX/AAXC files with vouchers and cover art
5. Decrypts to M4B (chapters preserved) using ffmpeg
6. Archives raw AAX files separately

## Quick start

```bash
# Build
docker build -t audible-backup .

# One-time auth (interactive — opens browser for Audible OAuth)
docker run -it -v ~/.audible:/root/.audible audible-backup \
  bash -c "audible quickstart"

# Dry run — see what's new without downloading
docker run --rm \
  -v ~/.audible:/root/.audible:ro \
  -v /path/to/audiobooks:/library \
  audible-backup --dry-run

# Full backup
docker run --rm \
  -v ~/.audible:/root/.audible:ro \
  -v /path/to/audiobooks:/library \
  audible-backup
```

## Library structure

The container expects an OpenAudible-compatible directory at `/library`:

```
/library/
  books.json                       # OpenAudible inventory (read-only, used for ASIN diffing)
  books/                           # Decoded audiobooks (M4B/MP3)
  aax/                             # Raw encrypted AAX files (archived after decrypt)
  art/                             # Cover art
  audible-backup-skiplist.txt      # Optional: ASINs to filter out before download
```

### Skip-list

Add an ASIN to `audible-backup-skiplist.txt` (one per line) to permanently filter it out of the download attempt. Use the `# comment` syntax to remember why:

```
1663704015  # Healing the Wounds — Plus-only, no longer in catalog
B079LRSMNN  # Galaxy's Edge — Plus-borrowed, never owned
```

The skip-list is the operator's explicit ack that an ASIN is intentionally absent. Without it, every weekly run re-attempts the same not-downloadable books and surfaces a fresh `status: "skipped"` record — noisy, and trains the operator to ignore the skip bucket entirely. ASINs filtered by the list appear in the JSON output's `skipped_by_filter` field so the audit trail is preserved.

If you don't have an existing OpenAudible library, create a minimal one:

```bash
mkdir -p /path/to/audiobooks/books
echo '[]' > /path/to/audiobooks/books.json
```

## Options

| Flag | Description |
|------|-------------|
| `--dry-run` | List new books without downloading |
| `--json` | Output results as JSON (for automation) |

## JSON output

With `--json`, the output is machine-readable:

```json
{
  "new_books": 5,
  "downloaded": 5,
  "skipped": 0,
  "failed": 0,
  "missing_on_disk": 1,
  "skipped_by_filter": ["1663704015", "B079LRSMNN"],
  "books": [
    {
      "asin": "B0AAAAAAAA",
      "title": "Some New Book",
      "author": "Jason Anspach",
      "narrated_by": "Mark Boyett",
      "status": "ok",
      "m4b_path": "/library/books/Some New Book.m4b"
    },
    {
      "asin": "ASIN999",
      "title": "Uncle Dynamite",
      "filename": "Uncle Dynamite.m4b",
      "status": "missing_on_disk"
    }
  ]
}
```

Field names match `backup.py`'s `map_to_inventory_schema()` output — `author` / `narrated_by` (singular), not `authors` / `narrators`. The full per-book record carries every field in `REQUIRED_OUTPUT_FIELDS`; only a subset is shown here. `status: "missing_on_disk"` records (counted under `missing_on_disk`) are inventory rows whose m4b file is no longer present under `/library/books/` — soft-alert only, no automatic redownload, lets the operator decide whether to re-fetch or accept the gap. `skipped_by_filter` lists ASINs from the operator-maintained skip-list (see "Library structure" above) that were filtered out before the download attempt — preserved at the top level so the audit trail survives runs where the filter prevents `books[]` from mentioning them.

## Scheduling

Run weekly via cron, systemd timer, or any scheduler:

```bash
0 3 * * 0  docker run --rm -v ~/.audible:/root/.audible:ro -v /mnt/audiobooks:/library audible-backup --json >> /var/log/audible-backup.log 2>&1
```

## Requirements

- Docker
- An Audible account (one-time interactive auth)
- An existing audiobook directory (OpenAudible format, or empty `books.json`)
