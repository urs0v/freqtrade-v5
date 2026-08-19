#!/usr/bin/env bash
set -euo pipefail

DB="${PREMIUM_DB:-/freqtrade/user_data/alpha_lab/premium_basis.sqlite}"
CONFIG="${PREMIUM_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${PREMIUM_DATADIR:-/freqtrade/user_data/data/binance}"
OUT="${PREMIUM_OUT:-/freqtrade/user_data/alpha_lab/premium_basis_alpha}"
WORKERS="${PREMIUM_WORKERS:-8}"

mkdir -p "$(dirname "$DB")" "$OUT"

echo "=== PREMIUM / BASIS LAB RUNNER ==="
echo "Backfill Binance premiumIndexKlines 15m, then run PTI-safe alpha gate."

python -u /opt/rmv5/tools/backfill_premium_basis.py \
  --config "$CONFIG" \
  --db "$DB" \
  --start "${PREMIUM_START:-2022-01-01}" \
  --end "${PREMIUM_END:-2026-08-19}" \
  --workers "$WORKERS"

python -u /opt/rmv5/tools/research_premium_basis_alpha.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --db "$DB" \
  --outdir "$OUT"
