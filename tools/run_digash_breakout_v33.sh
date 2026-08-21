#!/usr/bin/env bash
set -euo pipefail
EVENTS="${EVENTS:-/freqtrade/user_data/digash_breakout_v32/breakout_events_v32.csv}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/digash_breakout_v33}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
mkdir -p "$OUTDIR"
echo "=== DIGASH BREAKOUT V3.3 RUNNER ==="
echo "REPORT ONLY: consumes V3.2 events; no market downloads and no trade-rule changes."
echo "EVENTS=$EVENTS"
echo "OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/analyze_digash_breakout_v33.py \
  --events "$EVENTS" --outdir "$OUTDIR" --bootstrap "$BOOTSTRAP" 2>&1 | tee "$OUTDIR/run.log"
