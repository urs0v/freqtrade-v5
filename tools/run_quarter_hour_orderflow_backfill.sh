#!/usr/bin/env bash
set -euo pipefail

DB="${QH_DB:-/freqtrade/user_data/alpha_lab/qh_orderflow.sqlite}"
TMPDIR="${QH_TMPDIR:-/freqtrade/user_data/alpha_lab/qh_tmp}"
START="${QH_START:-2024-01-01}"
END="${QH_END:-2026-08-19}"
SYMBOLS="${QH_SYMBOLS:-BTCUSDT,ETHUSDT,XRPUSDT,SOLUSDT,DOGEUSDT,ADAUSDT}"
WORKERS="${QH_WORKERS:-2}"

mkdir -p "$(dirname "$DB")" "$TMPDIR"

python -u /opt/rmv5/tools/backfill_quarter_hour_orderflow.py \
  --db "$DB" \
  --tmpdir "$TMPDIR" \
  --start "$START" \
  --end "$END" \
  --symbols "$SYMBOLS" \
  --workers "$WORKERS"
