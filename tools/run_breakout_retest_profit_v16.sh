#!/usr/bin/env bash
set -euo pipefail

echo "=== BREAKOUT / RETEST PROFIT V1.6 RUNNER ==="
echo "Audits the frozen fakeout signal after removing future-within-bucket dedup. No tuning."

python /opt/rmv5/tools/breakout_retest_profit_v16.py \
  --config /freqtrade/user_data/v7/config-v7-core-backtest.json \
  --datadir /freqtrade/user_data/data/binance \
  --outdir /freqtrade/user_data/breakout_retest_profit_v16 \
  --start 2022-01-01 \
  --end 2026-08-19 \
  --workers 16

echo "=== DONE ==="
