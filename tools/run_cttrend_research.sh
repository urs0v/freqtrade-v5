#!/usr/bin/env bash
set -euo pipefail

EVAL_START="${1:-2022-01-01}"
EVAL_END="${2:-2026-07-31}"
DATA_START="${3:-2020-01-01}"
WORKERS="${WORKERS:-16}"
DB="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
OUT="/freqtrade/user_data/cttrend_research"

mkdir -p "$OUT"

echo "=== CTREND + 28D TSMOM BINANCE RESEARCH ==="
echo "Data backfill: $DATA_START -> $EVAL_END"
echo "Evaluation:    $EVAL_START -> $EVAL_END"
echo "Workers:       $WORKERS"
echo

echo "[1/2] Resume-aware historical Binance USD-M archive backfill (6h candles + funding)..."
python /opt/rmv5/tools/backfill_adaptivetrend_core_data_fast.py \
  --db "$DB" \
  --start "$DATA_START" \
  --end "$EVAL_END" \
  --workers "$WORKERS"

echo
echo "[2/2] Running pre-registered unbiased CTREND reconstruction..."
python /opt/rmv5/tools/cttrend_research_v2.py \
  --db "$DB" \
  --start "$EVAL_START" \
  --end "$EVAL_END" \
  --universe 50 \
  --top-frac 0.20 \
  --train-weeks 52 \
  --min-history-days 210 \
  --min-cross-section 15 \
  --side-cost-bps 7.0 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Full log: $OUT/run.log"
echo "CSV results: $OUT/weekly_results.csv, $OUT/year_breakdown.csv, $OUT/asset_contributions.csv"
echo "Paste the terminal section starting at '=== CTREND + 28D TSMOM: RESEARCH RESULT ===' into ChatGPT."
