#!/usr/bin/env bash
set -euo pipefail

# AyeAye verification for #497 (graceful-shutdown checkpoint pass).
# Runs every 10 min via cron. Each recent deploy-kill window should
# have at least one matching `<groupDir>/.checkpoints/default.md`
# whose mtime falls inside the kill window AND whose body carries the
# `**Trigger:** shutdown` marker the new code path renders. Alerts
# via Telegram if a kill window has no matching checkpoint.
#
# Why 10 min cadence: deploy-kills are rare (only on redeploy), but
# when they happen the next agent spawn happens within minutes. Catch
# the regression before the next user message rather than after a
# next-day audit.
#
# Cron: */10 * * * * /home/jbaruch/nanoclaw/scripts/verify-fixes/verify-497-shutdown-checkpoint.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="$ROOT_DIR/scripts/heartbeat-external.conf"
LOG_TAG="[$(date '+%Y-%m-%d %H:%M:%S')] verify-497"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "$LOG_TAG ERROR: config file not found: $CONFIG_FILE" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG_FILE"

KILL_LOG="$ROOT_DIR/data/host-logs/deploy-kills.log"
GROUPS_DIR="$ROOT_DIR/groups"

if [[ ! -f "$KILL_LOG" ]]; then
  echo "$LOG_TAG OK no deploy-kills.log yet"
  exit 0
fi

# Find deploy-kill windows in the last 24h. The log lines have ISO
# timestamps; we only care about the most recent kill since prior
# alerts already fired on older ones.
LATEST_KILL_LINE=$(tail -1 "$KILL_LOG" 2>/dev/null || true)
if [[ -z "$LATEST_KILL_LINE" ]]; then
  echo "$LOG_TAG OK kill log empty"
  exit 0
fi

# Parse the timestamp from the latest line. Format from
# `src/host-logs.ts` is "<ISO> killed <name>" — first field is ISO.
KILL_TS_ISO=$(echo "$LATEST_KILL_LINE" | awk '{print $1}')
KILL_TS_EPOCH=$(date -d "$KILL_TS_ISO" +%s 2>/dev/null || \
                date -j -f "%Y-%m-%dT%H:%M:%S%z" "$KILL_TS_ISO" +%s 2>/dev/null || \
                echo 0)

if [[ "$KILL_TS_EPOCH" -eq 0 ]]; then
  echo "$LOG_TAG WARN unable to parse kill timestamp: $LATEST_KILL_LINE" >&2
  exit 0
fi

# Only care about kills in the last 24h — older ones we've already
# alerted on and ignoring them avoids a stuck-alert loop.
NOW_EPOCH=$(date +%s)
AGE_SEC=$((NOW_EPOCH - KILL_TS_EPOCH))
if [[ "$AGE_SEC" -gt 86400 ]]; then
  echo "$LOG_TAG OK latest kill was ${AGE_SEC}s ago, outside 24h window"
  exit 0
fi

# Check every group's checkpoint. We expect at least ONE checkpoint
# whose mtime lands within ±120s of the kill (10s drain + buffer)
# AND whose body has the new "**Trigger:** shutdown" marker. The
# marker is the load-bearing signal — if it's missing, either the
# render path regressed or the checkpoint is from a threshold-cross
# (which is correct but unrelated to the SIGTERM path we're verifying).
WINDOW_LO=$((KILL_TS_EPOCH - 30))
WINDOW_HI=$((KILL_TS_EPOCH + 120))

matched=0
for cp in "$GROUPS_DIR"/*/.checkpoints/default.md; do
  [[ -f "$cp" ]] || continue
  mtime=$(stat -c '%Y' "$cp" 2>/dev/null || stat -f '%m' "$cp" 2>/dev/null || echo 0)
  if [[ "$mtime" -ge "$WINDOW_LO" && "$mtime" -le "$WINDOW_HI" ]]; then
    if grep -q '\*\*Trigger:\*\* shutdown' "$cp" 2>/dev/null; then
      matched=$((matched + 1))
    fi
  fi
done

echo "$LOG_TAG kill_at=$KILL_TS_ISO matched_checkpoints=$matched"

if [[ "$matched" -eq 0 ]]; then
  msg="🟡 <b>#497 regression suspected</b>"$'\n'$'\n'
  msg+="A deploy-kill happened at <code>$KILL_TS_ISO</code> but no "$'\n'
  msg+="checkpoint file under groups/*/.checkpoints/default.md "$'\n'
  msg+="has both the matching mtime window AND the "$'\n'
  msg+="<code>**Trigger:** shutdown</code> marker. Either the SIGTERM "$'\n'
  msg+="path skipped writeShutdownCheckpoints or the render regressed."$'\n'$'\n'
  msg+="<i>$(date '+%Y-%m-%d %H:%M:%S')</i>"
  curl -sf --max-time 15 \
    -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    -d "parse_mode=HTML" \
    --data-urlencode "text=${msg}" \
    -o /dev/null
  echo "$LOG_TAG ALERT: no matching shutdown-trigger checkpoint" >&2
fi
