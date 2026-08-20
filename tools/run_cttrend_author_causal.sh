#!/usr/bin/env bash
set -euo pipefail

EVAL_START="${1:-2022-01-01}"
EVAL_END="${2:-2026-07-31}"
CTREND_WORKERS="${CTREND_WORKERS:-64}"
DB="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
OUT="/freqtrade/user_data/cttrend_author_causal"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CTREND_WORKERS

mkdir -p "$OUT"

echo "=== CTREND AUTHOR-CODE CAUSAL FORENSIC ==="
echo "Evaluation:      $EVAL_START -> $EVAL_END"
echo "Workers:         $CTREND_WORKERS"
echo "DB:              $DB"
echo "No backfill. Reusing existing Binance candles + funding."
echo

python /opt/rmv5/tools/cttrend_author_causal.py \
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
echo "CSV: $OUT/weekly_results.csv, $OUT/rank_ic.csv, $OUT/year_breakdown.csv, $OUT/asset_contributions.csv"
echo "Paste the section starting at '=== AUTHOR-CODE-DERIVED CAUSAL RESULT ==='."
