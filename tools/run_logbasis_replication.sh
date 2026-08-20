#!/usr/bin/env bash
set -euo pipefail

START="${1:-2021-01-01}"
REP_END="${2:-2023-12-31}"
END="${3:-2026-07-31}"
WORKERS="${LOGBASIS_WORKERS:-48}"
CORE="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
DATA_DIR="/freqtrade/user_data/logbasis_8h"
DB="$DATA_DIR/logbasis.sqlite"
OUT="/freqtrade/user_data/logbasis_replication"

mkdir -p "$DATA_DIR" "$OUT"

echo "=== LOG-BASIS REPLICATION RUNNER ==="
echo "Replication era: $START -> $REP_END"
echo "Frozen OOS:      2024-01-01 -> $END"
echo "Data: official Binance USD-M + Spot 8h klines"
echo "Signal: ln(perp/spot), long lowest quintile / short highest quintile"
echo "Costs: 7bps per changed side; 10bps stress"
echo "No parameter search. Backfill is resumable."
echo

python /opt/rmv5/tools/backfill_logbasis_8h.py \
  --core "$CORE" \
  --db "$DB" \
  --start "$START" \
  --end "$END" \
  --workers "$WORKERS"

echo
python /opt/rmv5/tools/logbasis_replication.py \
  --basis-db "$DB" \
  --core-db "$CORE" \
  --start "$START" \
  --rep-end "$REP_END" \
  --end "$END" \
  --side-cost-bps 7.0 \
  --stress-side-cost-bps 10.0 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Log: $OUT/run.log"
echo "CSV: $OUT/summary.csv, $OUT/quintiles.csv, $OUT/year_breakdown.csv, $OUT/period_results.csv, $OUT/asset_results.csv, $OUT/gates.csv"
echo "Paste the block starting at '=== LOG-BASIS RESULT ==='."
