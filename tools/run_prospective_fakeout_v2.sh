#!/usr/bin/env bash
set -euo pipefail

OUTDIR="${OUTDIR:-/freqtrade/user_data/prospective_fakeout_v2}"
PIDFILE="$OUTDIR/tracker.pid"
LOGFILE="$OUTDIR/run.log"
PY="/opt/rmv5/tools/prospective_fakeout_v2.py"
PARITY="/opt/rmv5/tools/prospective_fakeout_v2_parity.py"
PARITY_JSON="$OUTDIR/parity_pass.json"
mkdir -p "$OUTDIR"

alive() {
  [[ -f "$PIDFILE" ]] || return 1
  local pid
  pid="$(cat "$PIDFILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

parity_ok() {
  [[ -f "$PARITY_JSON" ]] && grep -q '"verdict": "PARITY_PASS"' "$PARITY_JSON"
}

run_parity() {
  echo "=== PROSPECTIVE IMPLEMENTATION PARITY GATE ==="
  PYTHONUNBUFFERED=1 python -u "$PARITY" --outdir "$OUTDIR"
  if ! parity_ok; then
    echo "Parity gate did not pass. Prospective tracker will NOT start." >&2
    exit 2
  fi
}

cmd="${1:-start}"
case "$cmd" in
  start)
    if alive; then
      echo "Prospective tracker already running: pid=$(cat "$PIDFILE")"
      [[ -f "$OUTDIR/summary.txt" ]] && cat "$OUTDIR/summary.txt"
      exit 0
    fi
    if ! parity_ok; then
      run_parity
    else
      echo "Historical implementation parity: PASS"
    fi
    rm -f "$PIDFILE"
    nohup env PYTHONUNBUFFERED=1 python -u "$PY" --loop --outdir "$OUTDIR" >"$LOGFILE" 2>&1 &
    pid=$!
    echo "$pid" > "$PIDFILE"
    sleep 2
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "Tracker failed to start. Last log lines:"
      tail -n 80 "$LOGFILE" || true
      exit 1
    fi
    echo "Prospective tracker started: pid=$pid"
    echo "The prospective cutoff is created on this first start at the NEXT 5m boundary."
    echo "Log: $LOGFILE"
    echo "Report: bash /opt/rmv5/tools/run_prospective_fakeout_v2.sh report"
    echo "Status: bash /opt/rmv5/tools/run_prospective_fakeout_v2.sh status"
    ;;
  parity)
    run_parity
    ;;
  status)
    if alive; then
      echo "RUNNING pid=$(cat "$PIDFILE")"
    else
      echo "STOPPED"
    fi
    [[ -f "$PARITY_JSON" ]] && { echo "--- PARITY ---"; cat "$PARITY_JSON"; }
    [[ -f "$OUTDIR/state.json" ]] && { echo "--- STATE ---"; cat "$OUTDIR/state.json"; }
    [[ -f "$OUTDIR/summary.txt" ]] && { echo "--- SUMMARY ---"; cat "$OUTDIR/summary.txt"; }
    ;;
  report)
    if [[ -f "$OUTDIR/summary.txt" ]]; then
      cat "$OUTDIR/summary.txt"
    else
      echo "No summary yet. The first cycle starts after the next completed 5m candle."
      echo "Log tail:"
      tail -n 80 "$LOGFILE" 2>/dev/null || true
    fi
    ;;
  log)
    tail -n "${LINES:-120}" "$LOGFILE"
    ;;
  stop)
    if alive; then
      pid="$(cat "$PIDFILE")"
      kill "$pid" 2>/dev/null || true
      for _ in $(seq 1 20); do
        kill -0 "$pid" 2>/dev/null || break
        sleep 0.25
      done
      kill -9 "$pid" 2>/dev/null || true
      echo "Stopped pid=$pid"
    else
      echo "Already stopped"
    fi
    rm -f "$PIDFILE"
    ;;
  once)
    if ! parity_ok; then
      run_parity
    fi
    PYTHONUNBUFFERED=1 python -u "$PY" --outdir "$OUTDIR"
    ;;
  *)
    echo "Usage: $0 {start|parity|status|report|log|stop|once}" >&2
    exit 2
    ;;
esac
