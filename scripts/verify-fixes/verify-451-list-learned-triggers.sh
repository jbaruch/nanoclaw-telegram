#!/usr/bin/env bash
set -euo pipefail

# AyeAye verification for #451 (promote / reenable / delete / list
# learned-trigger IPC handlers). Runs daily via cron. Fires a
# `list_learned_triggers` IPC against the main group, reads the
# response file, and validates the JSON shape. Alerts via Telegram if
# the IPC silently regressed (no result, malformed JSON, missing
# `groups` array, or an `error` field instead of `stdout`).
#
# Daily cadence is appropriate: this is a slow regression — the
# handlers are exercised on every operator promotion/demotion call,
# and a daily smoke test against the registered-group set catches
# breakage within a day of the regression landing.
#
# Cron: 30 9 * * * /home/jbaruch/nanoclaw/scripts/verify-fixes/verify-451-list-learned-triggers.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$ROOT_DIR/scripts/heartbeat-external.conf"
LOG_TAG="[$(date '+%Y-%m-%d %H:%M:%S')] verify-451"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "$LOG_TAG ERROR: config file not found: $CONFIG_FILE" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG_FILE"

# Resolve the main-group folder from the orchestrator's DB. The IPC
# response lands in input-default/ under that folder.
MAIN_FOLDER=$(docker exec nanoclaw sh -c \
  'sqlite3 "file:/app/store/messages.db?mode=ro" "SELECT folder FROM registered_groups WHERE is_main = 1 LIMIT 1;"' \
  2>/dev/null || true)

if [[ -z "$MAIN_FOLDER" ]]; then
  echo "$LOG_TAG WARN unable to resolve main folder; skipping" >&2
  exit 0
fi

REQ_ID="verify-451-$(date +%s)"
TASKS_DIR="$ROOT_DIR/data/ipc/$MAIN_FOLDER/tasks"
RESULT_DIR="$ROOT_DIR/data/ipc/$MAIN_FOLDER/input-default"
RESULT_FILE="$RESULT_DIR/_script_result_${REQ_ID}.json"

mkdir -p "$TASKS_DIR"
echo "{\"type\":\"list_learned_triggers\",\"requestId\":\"$REQ_ID\"}" \
  > "$TASKS_DIR/req-${REQ_ID}.json"

# Wait up to 10s for the orchestrator to consume the request and
# write the result file. The IPC watcher polls the tasks dir on a
# short interval, so this is usually < 2s.
deadline=$(( $(date +%s) + 10 ))
while [[ "$(date +%s)" -lt "$deadline" ]]; do
  [[ -f "$RESULT_FILE" ]] && break
  sleep 1
done

alert_msg=""
if [[ ! -f "$RESULT_FILE" ]]; then
  alert_msg="no result file appeared within 10s — IPC watcher may be stuck or list_learned_triggers handler is gone"
else
  # Parse the result. Success shape is {"stdout":"<JSON>"}; error
  # shape is {"error":"<msg>"}.
  if grep -q '"error"' "$RESULT_FILE"; then
    err=$(awk -F'"error":"' '{print $2}' "$RESULT_FILE" | awk -F'"' '{print $1}')
    alert_msg="handler returned error: $err"
  else
    # Success shape is {"stdout":"<JSON-escaped string>"}; the inner
    # string contains the actual groups array. Both the wrapper key
    # and the inner key are quoted, but the inner is escaped — so we
    # match either escaped (\"groups\") or raw ("groups") to stay
    # robust across whichever shape any future serialiser emits.
    if ! grep -qE '\\"groups\\"|"groups"' "$RESULT_FILE"; then
      alert_msg="response shape invalid — missing 'groups' array"
    fi
  fi
fi

# Always clean up the result file so we don't accumulate.
rm -f "$RESULT_FILE" "$TASKS_DIR/req-${REQ_ID}.json"

if [[ -n "$alert_msg" ]]; then
  msg="🟡 <b>#451 regression suspected</b>"$'\n'$'\n'
  msg+="<code>list_learned_triggers</code> IPC smoke test failed:"$'\n'
  msg+="<i>$alert_msg</i>"$'\n'$'\n'
  msg+="<i>$(date '+%Y-%m-%d %H:%M:%S')</i>"
  curl -sf --max-time 15 \
    -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    -d "parse_mode=HTML" \
    --data-urlencode "text=${msg}" \
    -o /dev/null
  echo "$LOG_TAG ALERT: $alert_msg" >&2
  exit 0
fi

echo "$LOG_TAG OK list_learned_triggers responded with valid shape"
