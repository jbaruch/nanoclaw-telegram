#!/usr/bin/env bash
# Audible audiobook backup — downloads new books and decrypts to M4B.
# Reads OpenAudible's books.json as existing inventory, uses audible-cli for new purchases.
#
# Usage: ./scripts/audible-backup.sh [--dry-run]
#
# Prerequisites:
#   - audible-cli installed in ~/audible-env/ (python venv)
#   - ffmpeg available on PATH
#   - audible auth profile configured: ~/audible-env/bin/audible quickstart
#
# Environment:
#   AUDIOBOOK_DIR  — library root (default: /volume1/Google Drive/Audio Books)

set -euo pipefail

AUDIBLE="$HOME/audible-env/bin/audible"
AUDIOBOOK_DIR="${AUDIOBOOK_DIR:-/volume1/Google Drive/Audio Books}"
# OpenAudible names the file books.json but Google Drive sync may rename it
BOOKS_JSON=""
for candidate in "$AUDIOBOOK_DIR/books.json" "$AUDIOBOOK_DIR/books (1).json"; do
  if [ -f "$candidate" ]; then
    BOOKS_JSON="$candidate"
    break
  fi
done
BOOKS_DIR="$AUDIOBOOK_DIR/books"
AAX_DIR="$AUDIOBOOK_DIR/aax"
ART_DIR="$AUDIOBOOK_DIR/art"
DOWNLOAD_DIR="$AUDIOBOOK_DIR/tmp_download"
TMPDIR="/tmp/audible-backup-$$"

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=true
fi

cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

# --- Validate prerequisites ---
if [ ! -x "$AUDIBLE" ]; then
  echo "ERROR: audible-cli not found at $AUDIBLE"
  echo "Install: python3 -m venv ~/audible-env && ~/audible-env/bin/pip install audible-cli"
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ERROR: ffmpeg not found on PATH"
  exit 1
fi

if [ -z "$BOOKS_JSON" ]; then
  echo "ERROR: OpenAudible inventory not found in $AUDIOBOOK_DIR (tried books.json, books (1).json)"
  exit 1
fi

mkdir -p "$TMPDIR"

echo "=== Audible Backup ==="
echo "Library: $AUDIOBOOK_DIR"

# --- Fetch current Audible library ---
echo "Fetching Audible library..."
"$AUDIBLE" library export -f json -o "$TMPDIR/audible-library.json"

# --- Find new books ---
python3 << PYEOF
import json, os

inventory = json.load(open('$BOOKS_JSON'))
known_asins = {b.get('asin', '') for b in inventory if b.get('asin')}
print(f"Known books (OpenAudible): {len(known_asins)}")

library = json.load(open('$TMPDIR/audible-library.json'))
print(f"Audible library: {len(library)}")

new_books = []
for book in library:
    asin = book.get('asin', '')
    if asin and asin not in known_asins:
        new_books.append({
            'asin': asin,
            'title': book.get('title', 'Unknown'),
            'authors': book.get('authors', 'Unknown'),
        })

json.dump(new_books, open('$TMPDIR/new-books.json', 'w'), indent=2)
print(f"New books to download: {len(new_books)}")
for b in new_books:
    print(f"  {b['asin']}  {b['title']} — {b['authors']}")
PYEOF

NEW_COUNT=$(python3 -c "import json; print(len(json.load(open('$TMPDIR/new-books.json'))))")

if [ "$NEW_COUNT" -eq 0 ]; then
  echo "Library up to date."
  exit 0
fi

if [ "$DRY_RUN" = true ]; then
  echo ""
  echo "DRY RUN — nothing downloaded."
  exit 0
fi

# --- Download and decrypt each new book ---
mkdir -p "$DOWNLOAD_DIR" "$BOOKS_DIR" "$ART_DIR"

DOWNLOADED=0
FAILED=0

python3 -c "import json; [print(b['asin'], b['title'], sep='\t') for b in json.load(open('$TMPDIR/new-books.json'))]" | while IFS=$'\t' read -r ASIN TITLE; do
  echo ""
  echo "--- Downloading: $TITLE ($ASIN) ---"

  # Download AAX/AAXC + voucher + cover
  if ! "$AUDIBLE" download --asin "$ASIN" --output-dir "$DOWNLOAD_DIR" --cover --cover-size 500 --chapter 2>&1; then
    echo "FAILED to download $ASIN"
    FAILED=$((FAILED + 1))
    continue
  fi

  # Find the downloaded audio file
  AUDIO_FILE=$(find "$DOWNLOAD_DIR" -name "${ASIN}*" \( -name '*.aax' -o -name '*.aaxc' \) | head -1)
  VOUCHER_FILE=$(find "$DOWNLOAD_DIR" -name "${ASIN}*.voucher" | head -1)
  COVER_FILE=$(find "$DOWNLOAD_DIR" -name "${ASIN}*.jpg" -o -name "${ASIN}*.png" | head -1)

  if [ -z "$AUDIO_FILE" ]; then
    echo "FAILED: no audio file found for $ASIN after download"
    FAILED=$((FAILED + 1))
    continue
  fi

  # Sanitize title for filename
  SAFE_TITLE=$(echo "$TITLE" | sed 's/[^a-zA-Z0-9 ._-]//g' | sed 's/  */ /g' | head -c 200)
  OUTPUT_M4B="$BOOKS_DIR/$SAFE_TITLE.m4b"

  # Decrypt to M4B
  echo "Decrypting to M4B..."
  if [ -n "$VOUCHER_FILE" ]; then
    # AAXC format — needs voucher
    "$AUDIBLE" decrypt --input "$AUDIO_FILE" --voucher "$VOUCHER_FILE" --output "$OUTPUT_M4B" 2>&1
  else
    # AAX format — uses activation bytes from profile
    "$AUDIBLE" decrypt --input "$AUDIO_FILE" --output "$OUTPUT_M4B" 2>&1
  fi

  if [ -f "$OUTPUT_M4B" ]; then
    echo "OK: $SAFE_TITLE.m4b"
    DOWNLOADED=$((DOWNLOADED + 1))

    # Copy cover art
    if [ -n "$COVER_FILE" ] && [ -f "$COVER_FILE" ]; then
      cp "$COVER_FILE" "$ART_DIR/$SAFE_TITLE.jpg" 2>/dev/null
    fi

    # Archive raw AAX
    if [ -f "$AUDIO_FILE" ]; then
      mv "$AUDIO_FILE" "$AAX_DIR/" 2>/dev/null
    fi
  else
    echo "FAILED: decrypt produced no output for $ASIN"
    FAILED=$((FAILED + 1))
  fi

  # Clean up download temp files for this book
  find "$DOWNLOAD_DIR" -name "${ASIN}*" -delete 2>/dev/null
done

rmdir "$DOWNLOAD_DIR" 2>/dev/null

echo ""
echo "=== Backup complete ==="
echo "Downloaded: $DOWNLOADED"
echo "Failed: $FAILED"
