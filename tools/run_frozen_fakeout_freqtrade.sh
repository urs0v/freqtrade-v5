#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/opt/rmv5}"
USERDATA="${USERDATA:-/freqtrade/user_data}"
OUTDIR="${OUTDIR:-$USERDATA/frozen_fakeout_freqtrade}"
FEEDDIR="${FEEDDIR:-$USERDATA/frozen_fakeout_feed}"
CONFIG="${CONFIG:-$ROOT/config-frozen-fakeout.dryrun.json}"
STRATEGY_PATH="${STRATEGY_PATH:-$ROOT/strategies}"
STRATEGY="FrozenFakeoutV1"
DB_PATH="${DB_PATH:-$USERDATA/trades-frozen-fakeout.sqlite}"
DB_URL="${DB_URL:-sqlite:////freqtrade/user_data/trades-frozen-fakeout.sqlite}"
PARITY_JSON="$USERDATA/prospective_fakeout_v2/parity_pass.json"
PARITY_PY="$ROOT/tools/prospective_fakeout_v2_parity.py"
FEED_PY="$ROOT/tools/frozen_fakeout_signal_feed.py"
REPORT_PY="$ROOT/tools/frozen_fakeout_freqtrade_report.py"
FEED_PID="$OUTDIR/feed.pid"
BOT_PID="$OUTDIR/freqtrade.pid"
FEED_LOG="$OUTDIR/feed.log"
BOT_LOG="$OUTDIR/freqtrade.log"
mkdir -p "$OUTDIR" "$FEEDDIR"

alive_file() {
  local f="$1"
  [[ -f "$f" ]] || return 1
  local p
  p="$(cat "$f" 2>/dev/null || true)"
  [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null
}

parity_ok() {
  [[ -f "$PARITY_JSON" ]] && grep -q '"verdict": "PARITY_PASS"' "$PARITY_JSON"
}

preflight() {
  command -v freqtrade >/dev/null 2>&1 || { echo "freqtrade executable not found" >&2; return 1; }
  [[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; return 1; }
  [[ -f "$FEED_PY" ]] || { echo "Missing feed: $FEED_PY" >&2; return 1; }
  [[ -f "$REPORT_PY" ]] || { echo "Missing report helper: $REPORT_PY" >&2; return 1; }
  [[ -f "$STRATEGY_PATH/$STRATEGY.py" ]] || { echo "Missing strategy: $STRATEGY_PATH/$STRATEGY.py" >&2; return 1; }

  python -m py_compile "$FEED_PY" "$REPORT_PY" "$STRATEGY_PATH/$STRATEGY.py"
  python -m json.tool "$CONFIG" >/dev/null
  if ! freqtrade list-strategies --strategy-path "$STRATEGY_PATH" --no-color -1 2>&1 | grep -qx "$STRATEGY"; then
    echo "Freqtrade could not load $STRATEGY" >&2
    freqtrade list-strategies --strategy-path "$STRATEGY_PATH" --no-color 2>&1 || true
    return 1
  fi

  if ! parity_ok; then
    echo "Historical parity is not PASS; rerunning the parity gate before dry-run."
    PYTHONUNBUFFERED=1 python -u "$PARITY_PY" --outdir "$USERDATA/prospective_fakeout_v2"
  fi
  parity_ok || { echo "PARITY_PASS is required. Refusing to start Freqtrade." >&2; return 1; }
  echo "Preflight: strategy loads, config parses, historical parity=PASS."
}

stop_pidfile() {
  local f="$1"
  if alive_file "$f"; then
    local p
    p="$(cat "$f")"
    kill "$p" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$p" 2>/dev/null || break
      sleep 0.2
    done
    kill -9 "$p" 2>/dev/null || true
    echo "Stopped pid=$p"
  fi
  rm -f "$f"
}

start_all() {
  preflight
  if alive_file "$BOT_PID" || alive_file "$FEED_PID"; then
    echo "FrozenFakeout dry-run is already running."
    "$0" status
    return 0
  fi

  # Retire the older custom paper loop to avoid two full-history scanners using
  # CPU at once. Its files and original prospective cutoff are preserved, and
  # the executable feed inherits that cutoff.
  if [[ -f "$ROOT/tools/run_prospective_fakeout_v2.sh" ]]; then
    bash "$ROOT/tools/run_prospective_fakeout_v2.sh" stop >/dev/null 2>&1 || true
  fi

  rm -f "$FEED_PID" "$BOT_PID"
  nohup env PYTHONUNBUFFERED=1 python -u "$FEED_PY" \
    --loop --outdir "$FEEDDIR" >"$FEED_LOG" 2>&1 &
  local fp=$!
  echo "$fp" > "$FEED_PID"
  sleep 1
  if ! kill -0 "$fp" 2>/dev/null; then
    echo "Signal feed failed to start:" >&2
    tail -n 120 "$FEED_LOG" || true
    return 1
  fi

  nohup env FROZEN_FAKEOUT_FEED="$FEEDDIR/signals.csv" PYTHONUNBUFFERED=1 \
    freqtrade trade \
      --config "$CONFIG" \
      --strategy "$STRATEGY" \
      --strategy-path "$STRATEGY_PATH" \
      --db-url "$DB_URL" >"$BOT_LOG" 2>&1 &
  local bp=$!
  echo "$bp" > "$BOT_PID"
  sleep 3
  if ! kill -0 "$bp" 2>/dev/null; then
    echo "Freqtrade failed to start:" >&2
    tail -n 160 "$BOT_LOG" || true
    stop_pidfile "$FEED_PID"
    return 1
  fi

  echo "FrozenFakeoutV1 Freqtrade dry-run started."
  echo "feed pid=$fp | freqtrade pid=$bp"
  echo "feed=$FEEDDIR/signals.csv"
  echo "db=$DB_URL"
  echo "status: bash $ROOT/tools/run_frozen_fakeout_freqtrade.sh status"
  echo "report: bash $ROOT/tools/run_frozen_fakeout_freqtrade.sh report"
  echo "logs:   bash $ROOT/tools/run_frozen_fakeout_freqtrade.sh log"
}

cmd="${1:-start}"
case "$cmd" in
  start)
    start_all
    ;;
  preflight)
    preflight
    ;;
  status)
    if alive_file "$FEED_PID"; then echo "FEED RUNNING pid=$(cat "$FEED_PID")"; else echo "FEED STOPPED"; fi
    if alive_file "$BOT_PID"; then echo "FREQTRADE RUNNING pid=$(cat "$BOT_PID")"; else echo "FREQTRADE STOPPED"; fi
    [[ -f "$FEEDDIR/state.json" ]] && { echo "--- FEED STATE ---"; cat "$FEEDDIR/state.json"; }
    [[ -f "$FEEDDIR/snapshot.json" ]] && { echo "--- FEED SNAPSHOT ---"; cat "$FEEDDIR/snapshot.json"; }
    ;;
  report)
    python "$REPORT_PY" --db "$DB_PATH" --feed "$FEEDDIR"
    ;;
  log)
    echo "=== SIGNAL FEED LOG ==="
    tail -n "${LINES:-80}" "$FEED_LOG" 2>/dev/null || true
    echo
    echo "=== FREQTRADE LOG ==="
    tail -n "${LINES:-120}" "$BOT_LOG" 2>/dev/null || true
    ;;
  once)
    preflight
    PYTHONUNBUFFERED=1 python -u "$FEED_PY" --outdir "$FEEDDIR"
    ;;
  stop)
    stop_pidfile "$BOT_PID"
    stop_pidfile "$FEED_PID"
    echo "FrozenFakeout dry-run stopped. Database and feed files were preserved."
    ;;
  *)
    echo "Usage: $0 {start|preflight|status|report|log|once|stop}" >&2
    exit 2
    ;;
esac
