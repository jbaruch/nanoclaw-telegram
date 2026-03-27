#!/usr/bin/env bash
set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_FILE="${HOME}/Projects/nanoclaw/scripts/heartbeat-external.conf"
STATE_FILE="${HOME}/Projects/nanoclaw/scripts/heartbeat-external.state"
LOG_TAG="[$(date '+%Y-%m-%d %H:%M:%S')] heartbeat-external"

if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "$LOG_TAG ERROR: config file not found: $CONFIG_FILE" >&2
  exit 1
fi
# shellcheck source=/dev/null
source "$CONFIG_FILE"

# ── Helpers ───────────────────────────────────────────────────────────────────
PROBLEMS=()
WARNINGS=()

problem() { PROBLEMS+=("🔴 $1"); }
warn()    { WARNINGS+=("🟡 $1"); }

# Detect OS for stat flags
if [[ "$(uname)" == "Darwin" ]]; then
  stat_size() { stat -f '%z' "$1" 2>/dev/null; }
  stat_mtime() { stat -f '%m' "$1" 2>/dev/null; }
else
  stat_size() { stat -c '%s' "$1" 2>/dev/null; }
  stat_mtime() { stat -c '%Y' "$1" 2>/dev/null; }
fi

send_telegram() {
  local text="$1"
  curl -sf --max-time 15 \
    -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    -d "parse_mode=HTML" \
    --data-urlencode "text=${text}" \
    -o /dev/null
}

# Load state
last_db_size=0
if [[ -f "$STATE_FILE" ]]; then
  # shellcheck source=/dev/null
  source "$STATE_FILE"
  last_db_size="${last_db_size_bytes:-0}"
fi

# ── Check 7: Telegram reachability (do this first — needed for reporting) ─────
tg_status=$(curl -sf --max-time 10 \
  "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe" \
  -o /dev/null -w '%{http_code}' 2>/dev/null || echo "000")
telegram_reachable=true
if [[ "$tg_status" != "200" ]]; then
  telegram_reachable=false
  echo "$LOG_TAG ERROR: Telegram API unreachable (HTTP $tg_status)" >&2
fi

# ── Check 1: NanoClaw launchd service ─────────────────────────────────────────
if [[ "$(uname)" == "Darwin" ]]; then
  # macOS: check launchd
  if ! launchctl list com.nanoclaw &>/dev/null; then
    problem "NanoClaw: launchd service not loaded"
  else
    exit_status=$(launchctl list com.nanoclaw 2>/dev/null | grep 'LastExitStatus' | awk '{print $NF}' || echo "?")
    pid=$(launchctl list com.nanoclaw 2>/dev/null | grep '"PID"' | awk '{print $NF}' | tr -d ';' || echo "0")
    if [[ "$pid" == "0" || -z "$pid" ]]; then
      problem "NanoClaw: service loaded but not running (last exit: $exit_status)"
    fi
  fi
else
  # Linux/NAS: check if the orchestrator Docker container is running
  if docker ps --filter "name=nanoclaw" --filter "status=running" -q 2>/dev/null | grep -q .; then
    : # running
  elif pgrep -f 'dist/index.js' &>/dev/null; then
    : # running directly (non-Docker)
  else
    problem "NanoClaw: not running (neither container nor process found)"
  fi
fi

# ── Check 2: Agent containers ─────────────────────────────────────────────────
orphaned=$(docker ps -a \
  --filter "name=nanoclaw-" \
  --filter "status=exited" \
  --filter "status=dead" \
  --format '{{.Names}}' 2>/dev/null | wc -l | tr -d ' ')
if (( orphaned > 5 )); then
  problem "Orphaned containers: ${orphaned} stopped agent containers"
elif (( orphaned > 0 )); then
  warn "Orphaned containers: ${orphaned} stopped agent containers"
fi

running_agents=$(docker ps \
  --filter "name=nanoclaw-" \
  --format '{{.Names}}' 2>/dev/null | wc -l | tr -d ' ')
if (( running_agents > 10 )); then
  problem "Running agent containers: ${running_agents} (MAX_CONCURRENT is 5)"
fi

# ── Check 3: SQLite DB file ──────────────────────────────────────────────────
db_path="$NANOCLAW_DIR/store/messages.db"
if [[ ! -f "$db_path" ]]; then
  problem "Database: $db_path not found"
else
  db_size=$(stat_size "$db_path")
  db_mtime=$(stat_mtime "$db_path")
  now_epoch=$(date +%s)

  db_size_mb=$(( db_size / 1048576 ))
  if (( db_size_mb > 1024 )); then
    problem "Database: ${db_size_mb}MB (above 1GB)"
  elif (( db_size_mb > 500 )); then
    warn "Database: ${db_size_mb}MB (above 500MB)"
  fi

  stale_threshold=$(( DB_STALE_WARN_MINUTES * 60 ))
  seconds_since_write=$(( now_epoch - db_mtime ))
  minutes_since_write=$(( seconds_since_write / 60 ))
  if (( seconds_since_write > stale_threshold )); then
    warn "Database: last write ${minutes_since_write}m ago (stale?)"
  fi

  if (( last_db_size > 0 )); then
    growth_bytes=$(( db_size - last_db_size ))
    growth_mb=$(( growth_bytes / 1048576 ))
    if (( growth_mb > DB_GROWTH_WARN_MB )); then
      warn "Database: grew ${growth_mb}MB in last 15 minutes"
    fi
  fi
fi

# ── Check 4: Log files ───────────────────────────────────────────────────────
logs_dir="$NANOCLAW_DIR/logs"
if [[ -d "$logs_dir" ]]; then
  if [[ "$(uname)" == "Darwin" ]]; then
    logs_total=$(du -sm "$logs_dir" 2>/dev/null | awk '{print $1}')
  else
    logs_total=$(du -sm "$logs_dir" 2>/dev/null | awk '{print $1}')
  fi
  if (( ${logs_total:-0} > 200 )); then
    warn "Logs: ${logs_total}MB total"
  fi
  big_logs=$(find "$logs_dir" -type f -size +100M -name "*.log" 2>/dev/null | head -5)
  if [[ -n "$big_logs" ]]; then
    warn "Logs: files >100MB: $(echo "$big_logs" | tr '\n' ' ')"
  fi
fi

# ── Check 5: Disk space ─────────────────────────────────────────────────────
disk_pct=$(df "$NANOCLAW_DIR" 2>/dev/null | awk 'NR==2 {gsub(/%/,"",$5); print $5}')
if (( ${disk_pct:-0} >= 95 )); then
  problem "Disk: ${disk_pct}% used"
elif (( ${disk_pct:-0} >= DISK_WARN_THRESHOLD )); then
  warn "Disk: ${disk_pct}% used"
fi

# ── Check 6: Credential proxy health ─────────────────────────────────────────
proxy_status=$(curl -sf --max-time 5 \
  "http://localhost:3001/" \
  -o /dev/null -w '%{http_code}' 2>/dev/null || echo "000")
# Proxy returns 502 when no upstream — that means it's running
if [[ "$proxy_status" == "000" ]]; then
  warn "Credential proxy: not reachable on port 3001"
fi

# ── Check 7: Stuck scheduled tasks ───────────────────────────────────────────
if [[ -f "$db_path" ]] && command -v sqlite3 &>/dev/null; then
  stuck=$(sqlite3 "$db_path" "
    SELECT COUNT(*) FROM scheduled_tasks
    WHERE status='active'
      AND next_run <= datetime('now', '-5 minutes');
  " 2>/dev/null || echo 0)
  if (( ${stuck:-0} > 0 )); then
    # Auto-fix: reset next_run
    sqlite3 "$db_path" "
      UPDATE scheduled_tasks
      SET next_run = datetime('now', '+1 minute')
      WHERE status='active'
        AND next_run <= datetime('now', '-5 minutes');
    " 2>/dev/null
    warn "Stuck tasks: ${stuck} task(s) overdue — reset next_run"
  fi
fi

# ── Check 8: Session bloat + cleanup ─────────────────────────────────────────
sessions_dir="$NANOCLAW_DIR/data/sessions"
if [[ -d "$sessions_dir" ]]; then
  sessions_total=$(du -sm "$sessions_dir" 2>/dev/null | awk '{print $1}')
  if (( ${sessions_total:-0} > 500 )); then
    warn "Sessions: ${sessions_total}MB total (above 500MB)"
  fi
  # Auto-cleanup: delete session transcripts older than 7 days, keep latest 5
  cleaned=$(find "$sessions_dir" -path '*/.claude/projects/*/*.jsonl' -mtime +7 2>/dev/null | wc -l | tr -d ' ')
  if (( cleaned > 0 )); then
    find "$sessions_dir" -path '*/.claude/projects/*/*.jsonl' -mtime +7 -delete 2>/dev/null
    # Also clean subagent dirs
    find "$sessions_dir" -path '*/.claude/projects/*/subagents' -type d -empty -delete 2>/dev/null
    warn "Sessions: cleaned ${cleaned} old session files"
  fi
fi

# ── Check 9: Stuck IPC close files ───────────────────────────────────────────
stuck_close=$(find "$NANOCLAW_DIR/data/ipc" -name '_close' -mmin +30 2>/dev/null | wc -l | tr -d ' ')
if (( stuck_close > 0 )); then
  # Auto-fix: delete stuck close sentinels
  find "$NANOCLAW_DIR/data/ipc" -name '_close' -mmin +30 -delete 2>/dev/null
  warn "IPC: deleted ${stuck_close} stuck _close file(s)"
fi

# ── Check 10: Retry exhaustion ───────────────────────────────────────────────
log_file="$NANOCLAW_DIR/logs/nanoclaw.log"
if [[ -f "$log_file" ]]; then
  current_drops=$(grep -c 'Max retries exceeded' "$log_file" 2>/dev/null || echo 0)
  prev_drops="${last_retry_drops:-$current_drops}"
  new_drops=$(( current_drops - prev_drops ))
  if (( new_drops > 0 )); then
    warn "Retry exhaustion: ${new_drops} new dropped message(s)"
  fi
fi

# ── Check 11: Unanswered messages ────────────────────────────────────────────
if [[ -f "$db_path" ]] && command -v sqlite3 &>/dev/null; then
  unanswered=$(sqlite3 "$db_path" "
    SELECT COUNT(*) FROM messages m
    WHERE m.is_from_me = 0
      AND m.is_bot_message = 0
      AND m.timestamp >= datetime('now', '-15 minutes')
      AND m.timestamp <= datetime('now', '-5 minutes')
      AND NOT EXISTS (
        SELECT 1 FROM messages r
        WHERE r.chat_jid = m.chat_jid
          AND r.timestamp > m.timestamp
          AND r.is_bot_message = 1
      );
  " 2>/dev/null || echo 0)
  if (( ${unanswered:-0} > 0 )); then
    warn "Unanswered: ${unanswered} message(s) with no bot reply in 5-15 min"
  fi
fi

# ── Report ───────────────────────────────────────────────────────────────────
all_issues=("${PROBLEMS[@]+"${PROBLEMS[@]}"}" "${WARNINGS[@]+"${WARNINGS[@]}"}")

# Filter empty entries
filtered=()
for item in "${all_issues[@]}"; do
  [[ -n "$item" ]] && filtered+=("$item")
done

if (( ${#filtered[@]} == 0 )); then
  echo "$LOG_TAG OK"
  {
    echo "last_db_size_bytes=$(stat_size "$db_path" 2>/dev/null || echo 0)"
    echo "last_run_epoch=$(date +%s)"
    echo "last_retry_drops=$(grep -c 'Max retries exceeded' "$NANOCLAW_DIR/logs/nanoclaw.log" 2>/dev/null || echo 0)"
  } > "$STATE_FILE"
  exit 0
fi

# Build message
hostname_short=$(hostname -s 2>/dev/null || hostname)
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
msg="🫀 <b>External Heartbeat</b>"$'\n'
for issue in "${filtered[@]}"; do
  msg+=$'\n'"${issue}"
done
msg+=$'\n'$'\n'"<i>Host: ${hostname_short}  |  ${timestamp}</i>"

echo "$LOG_TAG ISSUES: ${#PROBLEMS[@]} problems, ${#WARNINGS[@]} warnings"

if $telegram_reachable; then
  send_telegram "$msg" || echo "$LOG_TAG ERROR: failed to send Telegram message" >&2
else
  echo "$LOG_TAG WARNING: Telegram unreachable, could not send alert" >&2
fi

{
  echo "last_db_size_bytes=$(stat_size "$db_path" 2>/dev/null || echo 0)"
  echo "last_run_epoch=$(date +%s)"
  echo "last_retry_drops=$(grep -c 'Max retries exceeded' "$NANOCLAW_DIR/logs/nanoclaw.log" 2>/dev/null || echo 0)"
} > "$STATE_FILE"

(( ${#PROBLEMS[@]} > 0 )) && exit 1 || exit 0
