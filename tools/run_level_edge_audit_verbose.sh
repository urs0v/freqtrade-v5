#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/level_edge_audit}"
START="${START:-2022-01-01}"
END="${END:-2026-08-19}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-10}"

if [[ ! -f "$CONFIG" ]]; then
  echo "CONFIG_MISSING: $CONFIG"
  exit 2
fi

mkdir -p "$OUTDIR"
LOG="$OUTDIR/run.log"

echo "=== LEVEL BOUNCE / BREAK-RETEST AUDIT ==="
echo "CACHE ONLY: this runner does not download market data."
echo "Python output is unbuffered; stage logs + heartbeat every ${HEARTBEAT_SEC}s."
echo "CONFIG=$CONFIG"
echo "DATADIR=$DATADIR"
echo "START=$START END=$END"
echo

START_TS=$(date +%s)

PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/audit_level_edge_verbose.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --outdir "$OUTDIR" \
  --start "$START" \
  --end "$END" \
  > >(tee "$LOG") 2>&1 &
PID=$!

heartbeat() {
  while kill -0 "$PID" 2>/dev/null; do
    sleep "$HEARTBEAT_SEC"
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    NOW=$(date +%s)
    ELAPSED=$((NOW-START_TS))
    LAST=$(tail -n 1 "$LOG" 2>/dev/null || true)
    echo "[heartbeat ${ELAPSED}s] still running | last: ${LAST:-waiting for first Python progress line}"
  done
}
heartbeat &
HB_PID=$!

set +e
wait "$PID"
RC=$?
set -e
kill "$HB_PID" 2>/dev/null || true
wait "$HB_PID" 2>/dev/null || true

echo
if [[ "$RC" -eq 0 ]]; then
  echo "=== DONE ==="
  echo "Paste from '=== SUMMARY: ACTIVE SUBSET / 8 BPS ===' through the end."
else
  echo "=== FAILED rc=$RC ==="
  echo "Last 40 log lines:"
  tail -n 40 "$LOG" || true
fi
exit "$RC"
