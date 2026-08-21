#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/breakout_retest_profit_v1}"
START="${START:-2022-01-01}"
END="${END:-2026-08-19}"
WORKERS="${WORKERS:-16}"
RISK_PCT="${RISK_PCT:-1.0}"
mkdir -p "$OUTDIR"
echo "=== BREAKOUT / RETEST PROFIT RESEARCH V1 RUNNER ==="
echo "PROFIT-FIRST, CACHE ONLY. No Digash fidelity/source matching."
echo "WORKERS=$WORKERS START=$START END=$END RISK_PCT=$RISK_PCT"
echo "OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/breakout_retest_profit_v1.py \
  --config "$CONFIG" --datadir "$DATADIR" --outdir "$OUTDIR" \
  --start "$START" --end "$END" --workers "$WORKERS" --risk-pct "$RISK_PCT" \
  "$@" 2>&1 | tee "$OUTDIR/run.log"
