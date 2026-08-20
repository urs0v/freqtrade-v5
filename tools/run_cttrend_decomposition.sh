#!/usr/bin/env bash
set -euo pipefail

EVAL_START="${1:-2022-01-01}"
EVAL_END="${2:-2026-07-31}"
CTREND_WORKERS="${CTREND_WORKERS:-32}"
DB="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
OUT="/freqtrade/user_data/cttrend_decomposition"

# Avoid BLAS/OpenMP oversubscription when many weekly fits run concurrently.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CTREND_WORKERS

mkdir -p "$OUT"

echo "=== CTREND DECOMPOSITION AUDIT ==="
echo "Evaluation:      $EVAL_START -> $EVAL_END"
echo "Model workers:   $CTREND_WORKERS"
echo "DB:              $DB"
echo "No backfill. Reusing the existing 2.4M-candle Binance archive."
echo

python /opt/rmv5/tools/cttrend_decomposition.py \
  --db "$DB" \
  --start "$EVAL_START" \
  --end "$EVAL_END" \
  --universe 50 \
  --top-frac 0.20 \
  --train-weeks 52 \
  --min-history-days 210 \
  --min-cross-section 15 \
  --side-cost-bps 7.0 \
  --workers "$CTREND_WORKERS" \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Full log: $OUT/run.log"
echo "CSV: $OUT/summary.csv, $OUT/rank_ic_by_week.csv, $OUT/year_breakdown.csv, $OUT/weekly_decomposition.csv"
echo "Paste the section starting at '=== CTREND DECOMPOSITION AUDIT ===' after the model run."
