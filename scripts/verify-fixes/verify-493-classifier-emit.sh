#!/usr/bin/env bash
set -euo pipefail

# AyeAye verification for #493 (host-side Haiku classifier emits to
# usage.jsonl). Runs hourly via cron. Alerts via Telegram if traffic
# arrived on non-main groups during the last hour but zero classifier
# records were captured — that's the regression mode the fix exists
# to prevent.
#
# Reads HOST paths directly (orchestrator side) — no container needed.
# This is a regression watchdog, not a one-shot smoke test; the live
# verification ran post-deploy on 2026-05-07 (PR #518).
#
# Cron: 5 * * * * /home/jbaruch/nanoclaw/scripts/verify-fixes/verify-493-classifier-emit.sh
#
# Sourcing the shared `heartbeat-external.conf` for TELEGRAM_BOT_TOKEN
# + TELEGRAM_CHAT_ID — same channel as the external heartbeat, so all
# regression alerts land in one chat.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$ROOT_DIR/scripts/heartbeat-external.conf"
LOG_TAG="[$(date '+%Y-%m-%d %H:%M:%S')] verify-493"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "$LOG_TAG ERROR: config file not found: $CONFIG_FILE" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG_FILE"

USAGE_LOG="$ROOT_DIR/logs/usage.jsonl"
STORE_DB="$ROOT_DIR/store/messages.db"

if [[ ! -f "$USAGE_LOG" ]]; then
  echo "$LOG_TAG WARN: usage.jsonl missing; orchestrator may not have started yet" >&2
  exit 0
fi
if [[ ! -f "$STORE_DB" ]]; then
  echo "$LOG_TAG WARN: messages.db missing" >&2
  exit 0
fi

# Window: last 60 minutes, ISO-8601 UTC.
WINDOW_START_EPOCH=$(($(date +%s) - 3600))
WINDOW_START=$(date -u -d "@$WINDOW_START_EPOCH" +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || \
               date -u -r "$WINDOW_START_EPOCH" +"%Y-%m-%dT%H:%M:%SZ")

# Count classifier-tier records in the last hour. The proxy-side
# capture and the host-side classifier both write to the same
# usage.jsonl, so we filter on the tier sentinel.
classifier_count=$(awk -v cutoff="$WINDOW_START" '
  /"tier":"classifier"/ {
    match($0, /"ts":"[^"]+"/)
    if (RSTART) {
      ts = substr($0, RSTART+6, RLENGTH-7)
      if (ts >= cutoff) c++
    }
  }
  END { print c+0 }
' "$USAGE_LOG")

# Count inbound messages on non-main groups in the last hour. The
# trigger gate's haiku classifier fires on every non-main inbound, so
# this is the lower-bound expected classifier-record count.
inbound_count=$(sqlite3 "file:$STORE_DB?mode=ro" \
  "SELECT COUNT(*) FROM messages m
   JOIN registered_groups g ON g.jid = m.chat_jid
   WHERE m.timestamp >= '$WINDOW_START'
     AND COALESCE(g.is_main, 0) = 0
     AND COALESCE(m.is_from_me, 0) = 0;" 2>/dev/null || echo 0)

echo "$LOG_TAG window=$WINDOW_START classifier=$classifier_count inbound_non_main=$inbound_count"

# Regression check: traffic arrived but no classifier records were
# captured. Either the gate is bypassed (config drift) or the emit
# from #493 silently regressed.
if [[ "$inbound_count" -gt 0 && "$classifier_count" -eq 0 ]]; then
  msg="🟡 <b>#493 regression suspected</b>"$'\n'$'\n'
  msg+="In the last hour, $inbound_count non-main inbound message(s) "$'\n'
  msg+="arrived but <b>0 classifier records</b> landed in usage.jsonl. "$'\n'
  msg+="Either the haiku-classifier gate stopped firing or the #493 "$'\n'
  msg+="emit hook regressed. Run:"$'\n'
  msg+="<code>tail -200 ~/nanoclaw/logs/usage.jsonl | grep classifier</code>"$'\n'$'\n'
  msg+="<i>$(date '+%Y-%m-%d %H:%M:%S')</i>"
  curl -sf --max-time 15 \
    -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    -d "parse_mode=HTML" \
    --data-urlencode "text=${msg}" \
    -o /dev/null
  echo "$LOG_TAG ALERT: classifier_count=0 with inbound_count=$inbound_count" >&2
fi
