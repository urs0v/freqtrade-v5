#!/usr/bin/env bash
set -euo pipefail

EVAL_START="${1:-2022-01-01}"
EVAL_END="${2:-2026-07-31}"
CTREND_WORKERS="${CTREND_WORKERS:-64}"
DB="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
OUT="/freqtrade/user_data/cttrend_market_neutral"

# One BLAS thread per concurrent weekly fit prevents oversubscription.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CTREND_WORKERS

mkdir -p "$OUT"

echo "=== CTREND MARKET-NEUTRAL SPREAD TEST ==="
echo "Evaluation:    $EVAL_START -> $EVAL_END"
echo "Model workers: $CTREND_WORKERS"
echo "DB:            $DB"
echo "No backfill. Reuses existing Binance candles/funding."
echo

python /opt/rmv5/tools/cttrend_market_neutral.py \
  --db "$DB" \
  --start "$EVAL_START" \
  --end "$EVAL_END" \
  --train-weeks 52 \
  --eval-universe 50 \
  --top-frac 0.20 \
  --min-cross-section 25 \
  --side-cost-bps 7.0 \
  --workers "$CTREND_WORKERS" \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Full log: $OUT/run.log"
echo "CSV: $OUT/summary.csv, $OUT/year_breakdown.csv, $OUT/weekly_spreads.csv, $OUT/ranking_monotonicity.csv, $OUT/signal_panel.csv"
echo "Paste the section starting at '=== CTREND SPREAD RESULT ==='."
