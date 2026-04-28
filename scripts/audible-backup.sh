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

# Per-title structured report (#147). The aggregate counters in the
# tail summary tell weekly housekeeping how MANY failed but lose every
# title-specific reason — operators and skills consuming the run output
# have to re-derive the per-title state from stdout. Write each
# per-title outcome as a single JSON object on its own line so a
# downstream consumer can parse with `jq -c .` or equivalent without
# heuristic regex on prose. Path is under the library root so it lives
# next to the artifacts the script produces (rather than a per-host
# /tmp path that vanishes on reboot).
#
# Schema (`schema_version: 1` per `jbaruch/coding-policy: stateful-artifacts`):
#   {"schema_version":1,"asin","title",
#    "stage":"download|classify|decrypt|copy|finalize|success",
#    "result":"failure|success","reason":"<string>","source_file":"<path>",
#    "output":"<path>","touched_files":["<path>",...]}
# Final line is a summary record (also schema_version: 1):
#   {"schema_version":1,"summary":true,"started_at":"...","ended_at":"...",
#    "downloaded":N,"failed":N}
REPORT_FILE="$AUDIOBOOK_DIR/.audible-backup-last-run.jsonl"
REPORT_SCHEMA_VERSION=1
REPORT_STARTED_AT=""

# Append a JSON record. Bash-only, no jq dep — careful field escaping
# so titles containing quotes or newlines don't break a downstream
# parse. All input goes through `python3 -c json.dumps` so quoting
# bugs (a backslash inside a title, an embedded newline) can't corrupt
# the JSONL stream the way printf %s would.
#
# Field encoding: values default to JSON strings. To emit a JSON
# number, boolean, or array, prefix the value with a type tag:
#   key=str      → "key": "str"  (default; tag-free)
#   key=int:5    → "key": 5
#   key=bool:true → "key": true
#   key=jsonl:a/b/c → "key": ["a","b","c"]   (NUL-free pipe-separated list — see TOUCHED_LIST below)
#                                              (caller chooses delimiter via the helper, here `/` for paths)
# Unknown type tags fall back to string (defense in depth — a typo
# emits "<value>" rather than a parse-fail; the schema_version on
# every record makes drift visible if the caller relied on a typed
# field that ended up stringified).
#
# Schema_version is auto-injected so callers can't forget it.
report_record() {
  python3 - "$REPORT_FILE" "$REPORT_SCHEMA_VERSION" "$@" <<'PYEOF'
import json, sys
report_path = sys.argv[1]
schema_version = int(sys.argv[2])
fields = sys.argv[3:]
record = {"schema_version": schema_version}
for entry in fields:
    if "=" not in entry:
        continue
    key, _, raw = entry.partition("=")
    if raw.startswith("int:"):
        try:
            record[key] = int(raw[4:])
        except ValueError:
            record[key] = raw
    elif raw.startswith("bool:"):
        record[key] = raw[5:].lower() == "true"
    elif raw.startswith("jsonl:"):
        # Pipe-separated list. Caller is responsible for picking a
        # delimiter that won't appear inside any value (file paths
        # never contain `\x1f` aka US, used here).
        body = raw[6:]
        record[key] = [p for p in body.split("\x1f") if p]
    else:
        record[key] = raw
with open(report_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PYEOF
}

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

# Truncate the per-run report — every invocation starts fresh so a
# stale prior run can't be confused with the current one. Done AFTER
# prerequisite checks so a hard-fail-before-the-loop preserves the
# previous run's report rather than blanking it.
: > "$REPORT_FILE"
REPORT_STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

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

# Materialize the ASIN/title pairs to a temp file BEFORE the loop, so
# a failure in the Python enumerator can be caught by set -e instead
# of silently producing an empty TSV (which would make the loop run
# zero iterations and the summary report "Downloaded: 0 / Failed: 0"
# as if everything was already up to date). `done < file` also keeps
# the loop in the current shell so `DOWNLOADED` / `FAILED` counters
# remain in scope for the final summary — same reason we used process
# substitution earlier, minus the exit-status blindness.
ASIN_TSV="$TMPDIR/asin-title.tsv"
python3 -c "import json; [print(b['asin'], b['title'], sep='\t') for b in json.load(open('$TMPDIR/new-books.json'))]" > "$ASIN_TSV"

while IFS=$'\t' read -r ASIN TITLE; do
  echo ""
  echo "--- Downloading: $TITLE ($ASIN) ---"

  # Record the download-start timestamp so we can identify files
  # produced by THIS audible-cli run regardless of their filename
  # convention. audible-cli's default output naming has drifted over
  # releases: some versions prefix with the ASIN, others prefix with
  # the title, some use the title AND a bitrate suffix
  # (e.g. `Red_Rising-LC_64_22050_stereo.aax`). A bare `-name "${ASIN}*"`
  # filter misses the title-prefix variants — this is how the
  # 2026-04-10 Moriarty/Dave Smith stragglers and the 2026-04-19
  # Red Rising 471MB straggler ended up stuck in tmp_download/ (the
  # source files WERE downloaded; the find below just couldn't see
  # them, so decrypt was skipped and cleanup missed them too).
  BEFORE_DOWNLOAD=$(date +%s)
  # Small safety margin against filesystem timestamp resolution and
  # clock jitter (the mtime can round to the second, and on some FSes
  # files briefly appear with a timestamp a hair before the issuing
  # process's recorded start). -1s makes the mtime>this-stamp filter
  # inclusive of anything that landed in the same second.
  BEFORE_DOWNLOAD=$((BEFORE_DOWNLOAD - 1))

  # Download AAX/AAXC + voucher + cover
  if ! "$AUDIBLE" download --asin "$ASIN" --output-dir "$DOWNLOAD_DIR" --cover --cover-size 500 --chapter 2>&1; then
    echo "FAILED to download $ASIN (audible-cli exit non-zero)"
    report_record \
      "asin=$ASIN" "title=$TITLE" \
      "stage=download" "result=failure" \
      "reason=audible-cli exit non-zero"
    FAILED=$((FAILED + 1))
    continue
  fi

  # Classify the downloaded files by mtime + extension. `find -newer`
  # against a touched reference file works on every find implementation
  # we might encounter (GNU, BSD, BusyBox) — `find -newermt "@<epoch>"`
  # is GNU-only. Platform-detect via uname so neither touch nor date
  # needs a stderr-suppressed fallback (per the no-error-suppression
  # rule).
  # Per-book error handling: any failure in touch/date/find must NOT
  # abort the whole run (set -euo pipefail would otherwise kill the
  # script mid-batch, losing the summary). On failure, log a clear
  # per-ASIN message, clean up REF_TS if it was created, bump FAILED,
  # and continue to the next book.
  REF_TS="$TMPDIR/ref-$BEFORE_DOWNLOAD"
  rm -f "$REF_TS"
  case "$(uname -s)" in
    Darwin|*BSD|DragonFly)
      # BSD touch needs `-t YYYYMMDDHHMM.SS`, and BSD `date -r` reads
      # the epoch from its argument directly. DragonFlyBSD is listed
      # explicitly because its `uname -s` reports `DragonFly` (no `BSD`
      # suffix), which would otherwise fall through to the GNU branch
      # and fail on `touch -d "@epoch"`.
      if ! REF_TOUCH_TS="$(date -r "$BEFORE_DOWNLOAD" +%Y%m%d%H%M.%S)"; then
        echo "FAILED to classify downloaded files for $ASIN (could not format reference timestamp)"
        report_record "asin=$ASIN" "title=$TITLE" "stage=classify" "result=failure" "reason=could not format BSD reference timestamp"
        FAILED=$((FAILED + 1))
        rm -f "$REF_TS"
        continue
      fi
      if ! touch -t "$REF_TOUCH_TS" "$REF_TS"; then
        echo "FAILED to classify downloaded files for $ASIN (could not create reference timestamp file)"
        report_record "asin=$ASIN" "title=$TITLE" "stage=classify" "result=failure" "reason=BSD touch -t failed"
        FAILED=$((FAILED + 1))
        rm -f "$REF_TS"
        continue
      fi
      ;;
    *)
      # GNU touch (Linux, Synology NAS default) supports `-d "@epoch"`.
      # BusyBox will fail here visibly — which is the right behavior,
      # since the deploy target is GNU coreutils.
      if ! touch -d "@$BEFORE_DOWNLOAD" "$REF_TS"; then
        echo "FAILED to classify downloaded files for $ASIN (could not create reference timestamp file)"
        report_record "asin=$ASIN" "title=$TITLE" "stage=classify" "result=failure" "reason=GNU touch -d @epoch failed"
        FAILED=$((FAILED + 1))
        rm -f "$REF_TS"
        continue
      fi
      ;;
  esac
  if ! NEW_FILES=$(find "$DOWNLOAD_DIR" -type f -newer "$REF_TS" | sort); then
    echo "FAILED to classify downloaded files for $ASIN (find/sort error)"
    report_record "asin=$ASIN" "title=$TITLE" "stage=classify" "result=failure" "reason=find/sort error"
    FAILED=$((FAILED + 1))
    rm -f "$REF_TS"
    continue
  fi

  AUDIO_FILE=""
  SOURCE_KIND=""   # aax | aaxc | mp3 (unencrypted) | ""
  VOUCHER_FILE=""
  COVER_FILE=""
  # Classifier uses explicit priority rather than "last write wins."
  # Priority: aaxc (encrypted w/ voucher) > aax (encrypted w/ activation
  # bytes) > mp3 (unencrypted). Without this, a mixed payload where
  # audible-cli drops BOTH .aaxc and .aax for the same title could land
  # the wrong format depending on sort order / filenames, which in turn
  # would pick the wrong decrypt path. The priority scale is 3/2/1 so
  # any "higher" match upgrades SOURCE_KIND, same-level stays on first
  # match (deterministic via the `sort` above).
  _prio() { case "$1" in aaxc) echo 3;; aax) echo 2;; mp3) echo 1;; *) echo 0;; esac; }
  current_prio=0
  while IFS= read -r f; do
    [ -z "$f" ] && continue
    case "$f" in
      *.aaxc) new_kind="aaxc" ;;
      *.aax)  new_kind="aax" ;;
      # audible-cli can deliver an unencrypted MP3 for some titles
      # (short-form content, legacy free items). Treat as a distinct
      # source kind — decrypt doesn't apply; we just copy/rename the
      # file into BOOKS_DIR as a .mp3. The Dave Smith straggler in
      # 2026-04-10 hit exactly this path and got stuck because the old
      # classifier didn't recognise .mp3 at all.
      *.mp3)  new_kind="mp3" ;;
      *.voucher) VOUCHER_FILE="$f"; continue ;;
      *.jpg|*.jpeg|*.png) COVER_FILE="$f"; continue ;;
      *) continue ;;
    esac
    new_prio=$(_prio "$new_kind")
    if [ "$new_prio" -gt "$current_prio" ]; then
      AUDIO_FILE="$f"
      SOURCE_KIND="$new_kind"
      current_prio="$new_prio"
    fi
  done <<< "$NEW_FILES"

  if [ -z "$AUDIO_FILE" ]; then
    # Genuinely nothing downloaded — audible-cli exited 0 but produced
    # no recognizable audio artifact. List what DID land so the
    # operator can see the real state (empty, or an unexpected
    # extension we don't classify yet).
    echo "FAILED: no source audio file from audible-cli for $ASIN"
    echo "  files touched in this run:"
    # Iterate line-by-line — don't use `printf '%s\n' $NEW_FILES`
    # because unquoted expansion word-splits and glob-expands. A
    # filename containing spaces or a wildcard char would otherwise
    # be mangled or (worse) match other paths on disk.
    # Build the touched-files list as a JSON array via the
    # `jsonl:`-prefixed record helper. Use US (\x1f, ASCII 31) as the
    # delimiter — paths cannot contain control bytes, so the split is
    # unambiguous. Avoids the literal `\n` bug a quoted "${...}\n"
    # concatenation produces (double-quoted `\n` is two characters,
    # not a newline; downstream JSON would carry "foo\\nbar" and
    # confuse anyone trying to split on real newlines).
    TOUCHED_LIST=""
    while IFS= read -r _touched; do
      [ -z "$_touched" ] && continue
      printf '    %s\n' "$_touched"
      TOUCHED_LIST="${TOUCHED_LIST}${_touched}"$'\x1f'
    done <<< "$NEW_FILES"
    report_record \
      "asin=$ASIN" "title=$TITLE" "stage=classify" "result=failure" \
      "reason=no recognizable audio artifact (aax/aaxc/mp3) in tmp_download" \
      "touched_files=jsonl:$TOUCHED_LIST"
    FAILED=$((FAILED + 1))
    continue
  fi

  # Sanitize title for filename
  SAFE_TITLE=$(echo "$TITLE" | sed 's/[^a-zA-Z0-9 ._-]//g' | sed 's/  */ /g' | head -c 200)

  case "$SOURCE_KIND" in
    aaxc)
      OUTPUT_M4B="$BOOKS_DIR/$SAFE_TITLE.m4b"
      if [ -z "$VOUCHER_FILE" ]; then
        echo "FAILED: AAXC source for $ASIN requires a .voucher but none was downloaded"
        echo "  source file left in tmp_download for retry: $AUDIO_FILE"
        report_record \
          "asin=$ASIN" "title=$TITLE" "stage=decrypt" "result=failure" \
          "reason=AAXC source missing .voucher" \
          "source_file=$AUDIO_FILE"
        FAILED=$((FAILED + 1))
        continue
      fi
      echo "Decrypting AAXC (with voucher) to $SAFE_TITLE.m4b ..."
      if ! "$AUDIBLE" decrypt --input "$AUDIO_FILE" --voucher "$VOUCHER_FILE" --output "$OUTPUT_M4B"; then
        echo "FAILED: audible decrypt (aaxc) exit non-zero for $ASIN"
        echo "  source: $AUDIO_FILE (retained in tmp_download for retry)"
        report_record \
          "asin=$ASIN" "title=$TITLE" "stage=decrypt" "result=failure" \
          "reason=audible decrypt (aaxc) exit non-zero" \
          "source_file=$AUDIO_FILE"
        FAILED=$((FAILED + 1))
        continue
      fi
      ;;
    aax)
      OUTPUT_M4B="$BOOKS_DIR/$SAFE_TITLE.m4b"
      echo "Decrypting AAX (activation bytes) to $SAFE_TITLE.m4b ..."
      if ! "$AUDIBLE" decrypt --input "$AUDIO_FILE" --output "$OUTPUT_M4B"; then
        echo "FAILED: audible decrypt (aax) exit non-zero for $ASIN"
        echo "  source: $AUDIO_FILE (retained in tmp_download for retry)"
        echo "  hint: verify ~/.audible/config.toml has activation_bytes set for the active profile"
        report_record \
          "asin=$ASIN" "title=$TITLE" "stage=decrypt" "result=failure" \
          "reason=audible decrypt (aax) exit non-zero — check activation_bytes in ~/.audible/config.toml" \
          "source_file=$AUDIO_FILE"
        FAILED=$((FAILED + 1))
        continue
      fi
      ;;
    mp3)
      # Non-encrypted source. Preserve the .mp3 extension so downstream
      # tagging/players don't assume an MP4 container. OpenAudible's
      # index doesn't require .m4b uniformly — .mp3 is fine.
      OUTPUT_M4B="$BOOKS_DIR/$SAFE_TITLE.mp3"
      echo "Unencrypted MP3 source for $ASIN — copying as-is to $SAFE_TITLE.mp3"
      if ! cp "$AUDIO_FILE" "$OUTPUT_M4B"; then
        echo "FAILED: cp mp3 for $ASIN (destination $OUTPUT_M4B)"
        report_record \
          "asin=$ASIN" "title=$TITLE" "stage=copy" "result=failure" \
          "reason=cp mp3 failed" \
          "source_file=$AUDIO_FILE" "output=$OUTPUT_M4B"
        FAILED=$((FAILED + 1))
        continue
      fi
      ;;
  esac

  if [ -f "$OUTPUT_M4B" ]; then
    echo "OK: $(basename "$OUTPUT_M4B")"
    report_record \
      "asin=$ASIN" "title=$TITLE" "stage=success" "result=success" \
      "source_kind=$SOURCE_KIND" "output=$OUTPUT_M4B"
    DOWNLOADED=$((DOWNLOADED + 1))

    # Copy cover art. Preserve the original extension — some titles
    # ship .png or .jpeg, and renaming to .jpg without re-encoding
    # leaves consumers that sniff by extension confused (or worse,
    # players that fail silently on a mismatched container). Extract
    # the extension from the downloaded filename.
    #
    # set -e guard: cover copy failure is non-fatal per-book. Without
    # the `if ! cp ...` wrapper, a disk-full / permission / path-too-
    # long error would abort the whole run mid-loop, skipping the
    # final summary and leaving the book's decrypted m4b in BOOKS_DIR
    # without its art but with nothing logged. Explicit failure
    # handling keeps the loop moving.
    if [ -n "$COVER_FILE" ] && [ -f "$COVER_FILE" ]; then
      cover_ext="${COVER_FILE##*.}"
      if ! cp "$COVER_FILE" "$ART_DIR/$SAFE_TITLE.$cover_ext"; then
        echo "WARN: cover art copy failed for $ASIN — continuing without cover"
      fi
    fi

    # Archive raw source. AAX goes to the existing archive dir; AAXC
    # goes there too (so the voucher pairing is co-located if a future
    # re-decrypt is needed); MP3 sources stay in tmp_download until
    # cleanup because they're not intermediate — the destination copy
    # IS the deliverable.
    #
    # set -e guard: archive-mv failure is non-fatal per-book (same
    # rationale as the cover copy above). Log the failure; the raw
    # source stays in tmp_download and the mtime cleanup at the end
    # of the loop iteration will pick it up anyway, so we don't leak.
    if [ "$SOURCE_KIND" != "mp3" ] && [ -f "$AUDIO_FILE" ]; then
      mkdir -p "$AAX_DIR"
      if ! mv "$AUDIO_FILE" "$AAX_DIR/"; then
        # Archive is book-keeping: the m4b deliverable is already in
        # BOOKS_DIR, so the run is fundamentally a success. The AAX/
        # AAXC source will be swept by the mtime cleanup below — mild
        # loss for re-decrypt scenarios (only relevant for AAXC, where
        # the voucher + source would let you redo decryption without
        # re-downloading). Acceptable edge: archive mv failures are
        # rare (mkdir AAX_DIR already succeeded; usual cause is a
        # filesystem permission flip between the two dirs).
        echo "WARN: archive mv failed for $AUDIO_FILE — source will be swept by cleanup; m4b is already in BOOKS_DIR"
      fi
      # Co-locate the voucher with the AAXC archive copy.
      if [ "$SOURCE_KIND" = "aaxc" ] && [ -n "$VOUCHER_FILE" ] && [ -f "$VOUCHER_FILE" ]; then
        if ! mv "$VOUCHER_FILE" "$AAX_DIR/"; then
          echo "WARN: voucher mv failed for $VOUCHER_FILE — continuing"
        fi
      fi
    fi
  else
    echo "FAILED: decrypt/copy produced no output for $ASIN (expected $OUTPUT_M4B)"
    echo "  source retained in tmp_download for inspection: $AUDIO_FILE"
    report_record \
      "asin=$ASIN" "title=$TITLE" "stage=finalize" "result=failure" \
      "reason=decrypt/copy produced no output (expected $OUTPUT_M4B)" \
      "source_file=$AUDIO_FILE" "output=$OUTPUT_M4B"
    FAILED=$((FAILED + 1))
    # Skip the mtime-based cleanup below — the decrypt path claimed
    # success but produced no m4b, which is a weird state the operator
    # may want to debug with the source AAX/AAXC still in place.
    # Audible-cli will re-download next weekly run (OpenAudible's
    # books.json has no m4b_path for this ASIN → treats it as new).
    continue
  fi

  # Clean up this book's leftover files in tmp_download by mtime, not
  # by ASIN prefix. Catches the title-prefix filenames that the old
  # ASIN-prefix cleanup missed, so stragglers don't accumulate across
  # weekly runs. Reuse the same reference file from the classifier
  # (-newer vs -newermt) for portable find behavior.
  #
  # set -e guard: a permission issue on one stale file in tmp_download
  # would otherwise abort the whole run via find's non-zero exit, with
  # the still-open loop iteration blocking all subsequent books AND
  # the final summary line. Wrap in `|| true` so cleanup is genuinely
  # best-effort — a failed delete leaves the straggler for the next
  # run to try again, which is strictly safer than a hard abort.
  # Log the failure so systematic issues (tmp_download permissions
  # flipped, disk full, mtime ref lost) surface for the operator.
  if ! find "$DOWNLOAD_DIR" -type f -newer "$REF_TS" -delete; then
    echo "WARN: tmp_download cleanup partially failed for $ASIN — stragglers may retry next run"
  fi
  rm -f "$REF_TS"
done < "$ASIN_TSV"

rmdir "$DOWNLOAD_DIR" 2>/dev/null

echo ""
echo "=== Backup complete ==="
echo "Downloaded: $DOWNLOADED"
echo "Failed: $FAILED"

# Final summary record. Always written, even if the loop body never
# executed (NEW_COUNT=0 already exited above, but a future caller that
# trips a different early-out path benefits from a deterministic
# "summary always present" contract). Schema documented at the top of
# this script next to REPORT_FILE. Counts use the `int:` type tag so
# they land in the JSON as numbers (not strings) — downstream
# consumers can `>= N` directly. `summary` is a boolean for the same
# reason.
report_record \
  "summary=bool:true" \
  "started_at=$REPORT_STARTED_AT" \
  "ended_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "downloaded=int:$DOWNLOADED" \
  "failed=int:$FAILED"
echo "Per-title report: $REPORT_FILE"
