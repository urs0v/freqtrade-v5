#!/usr/bin/env bash
set -euo pipefail

START="${1:-2022-01-01}"
END="${2:-$(date -u -d 'yesterday' +%F)}"
LIVE_CONFIG="${RMV5_CONFIG:-/freqtrade/user_data/config-v5.generated.json}"
TEST_CONFIG="${ADAPTIVE_TEST_CONFIG:-/freqtrade/user_data/v5/config-adaptive20x-backtest.json}"
SCHEDULE="${ADAPTIVE_SCHEDULE:-/freqtrade/user_data/v5/adaptive-schedule.json}"
STRATEGY_PATH="${RMV5_STRATEGY_PATH:-/opt/rmv5/strategies}"
DATA_ROOT="${ADAPTIVE_DATA_ROOT:-/freqtrade/user_data/data/binance/futures}"
WALLET="${ADAPTIVE_BACKTEST_WALLET:-100}"
MAX_TRADES="${ADAPTIVE_MAX_OPEN_TRADES:-20}"
RANKING="${ADAPTIVE_RANKING:-coingecko}"
EMERGENCY_STOP="${ADAPTIVE_EMERGENCY_PRICE_STOP:-0.035}"

START_FT="${START//-/}"
END_FT="${END//-/}"
TIMERANGE="${START_FT}-${END_FT}"
CALIB_START="$(python - "$START" <<'PY'
import sys
from datetime import datetime, timedelta
x = datetime.strptime(sys.argv[1], '%Y-%m-%d') - timedelta(days=45)
print(x.strftime('%Y-%m-%d'))
PY
)"
CALIB_START_FT="${CALIB_START//-/}"
DOWNLOAD_RANGE="${CALIB_START_FT}-${END_FT}"

mkdir -p /freqtrade/user_data/v5

echo "=== 1/4 ENSURE H6 + 5M FUTURES DATA ==="
echo "Calibration data starts: $CALIB_START"
freqtrade download-data \
  --config "$LIVE_CONFIG" \
  --trading-mode futures \
  --timeframes 5m 6h \
  --timerange "$DOWNLOAD_RANGE"

echo "=== 2/4 BUILD MONTHLY WALK-FORWARD SCHEDULE ==="
python /opt/rmv5/tools/build_adaptive_schedule.py \
  --config "$LIVE_CONFIG" \
  --data-root "$DATA_ROOT" \
  --start "$START" \
  --end "$END" \
  --output "$SCHEDULE" \
  --ranking "$RANKING" \
  --fee 0.0004

echo "=== 3/4 BUILD 20X BACKTEST CONFIG ==="
python - "$LIVE_CONFIG" "$TEST_CONFIG" "$WALLET" "$MAX_TRADES" <<'PY'
import json, sys
src, dst, wallet, max_trades = sys.argv[1:]
cfg = json.load(open(src))
cfg["timeframe"] = "6h"
cfg["stake_amount"] = "unlimited"
cfg["tradable_balance_ratio"] = 0.99
cfg["dry_run_wallet"] = float(wallet)
cfg["max_open_trades"] = int(max_trades)
cfg["liquidation_buffer"] = max(float(cfg.get("liquidation_buffer", 0.15)), 0.15)
with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
print("AdaptiveTrend universe:", ", ".join(cfg["exchange"].get("pair_whitelist", [])))
print(f"wallet={wallet} USDT | max_open_trades={max_trades} | timeframe=6h | leverage=20x")
print("monthly sizing: 70% long leg / selected longs, 30% short leg / selected shorts")
PY

echo "=== 4/4 FULL ADAPTIVETREND 20X BACKTEST ==="
ADAPTIVE_SCHEDULE="$SCHEDULE" \
ADAPTIVE_FORCE_20X=true \
ADAPTIVE_EMERGENCY_PRICE_STOP="$EMERGENCY_STOP" \
freqtrade backtesting \
  --config "$TEST_CONFIG" \
  --strategy AdaptiveTrend20x \
  --strategy-path "$STRATEGY_PATH" \
  --timerange "$TIMERANGE" \
  --timeframe-detail 5m \
  --fee 0.0004 \
  --breakdown month year
