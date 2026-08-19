#!/usr/bin/env bash
set -euo pipefail

START="${1:-2026-01-01}"
END="${2:-$(date -u -d 'yesterday' +%F)}"
LIVE_CONFIG="${RMV5_CONFIG:-/freqtrade/user_data/config-v5.generated.json}"
TEST_CONFIG="${RMV6_TEST_CONFIG:-/freqtrade/user_data/v5/config-v6-backtest.json}"
STRATEGY_PATH="${RMV5_STRATEGY_PATH:-/opt/rmv5/strategies}"
WALLET="${RMV6_BACKTEST_WALLET:-100}"
MAX_TRADES="${RMV6_MAX_OPEN_TRADES:-5}"
LEVERAGE="${RMV6_LEVERAGE:-10}"
ACCOUNT_RISK="${RMV6_ACCOUNT_RISK:-0.03}"
COLLATERAL_CAP="${RMV6_COLLATERAL_CAP:-0.20}"
ATR_STOP_MULT="${RMV6_ATR_STOP_MULT:-2.2}"

START_FT="${START//-/}"
END_FT="${END//-/}"
TIMERANGE="${START_FT}-${END_FT}"
DOWNLOAD_START="$(python - "$START" <<'PY'
import sys
from datetime import datetime, timedelta
x = datetime.strptime(sys.argv[1], '%Y-%m-%d') - timedelta(days=60)
print(x.strftime('%Y-%m-%d'))
PY
)"
DOWNLOAD_START_FT="${DOWNLOAD_START//-/}"
DOWNLOAD_RANGE="${DOWNLOAD_START_FT}-${END_FT}"

mkdir -p /freqtrade/user_data/v5

echo "=== 1/3 ENSURE 1H + 4H + 5M FUTURES DATA ==="
echo "Data warmup starts: $DOWNLOAD_START"
freqtrade download-data \
  --config "$LIVE_CONFIG" \
  --trading-mode futures \
  --timeframes 5m 1h 4h \
  --timerange "$DOWNLOAD_RANGE"

echo "=== 2/3 BUILD V6 BACKTEST CONFIG ==="
python - "$LIVE_CONFIG" "$TEST_CONFIG" "$WALLET" "$MAX_TRADES" <<'PY'
import json, sys
src, dst, wallet, max_trades = sys.argv[1:]
cfg = json.load(open(src))
cfg["timeframe"] = "1h"
cfg["stake_amount"] = "unlimited"
cfg["tradable_balance_ratio"] = 0.99
cfg["dry_run_wallet"] = float(wallet)
cfg["max_open_trades"] = int(max_trades)
cfg["liquidation_buffer"] = max(float(cfg.get("liquidation_buffer", 0.15)), 0.15)
with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
print("V6 universe:", ", ".join(cfg["exchange"].get("pair_whitelist", [])))
PY

echo "=== 3/3 REGIMEMOMENTUM V6 BACKTEST ==="
echo "Period: $START -> $END"
echo "wallet=$WALLET | leverage=${LEVERAGE}x | max_open_trades=$MAX_TRADES"
echo "risk/trade=${ACCOUNT_RISK} equity | collateral cap=${COLLATERAL_CAP} equity | ATR stop mult=$ATR_STOP_MULT"
echo "Primary evaluation: MONTHLY return distribution, PF, drawdown, long/short split, outlier dependence."

RMV6_LEVERAGE="$LEVERAGE" \
RMV6_ACCOUNT_RISK="$ACCOUNT_RISK" \
RMV6_COLLATERAL_CAP="$COLLATERAL_CAP" \
RMV6_ATR_STOP_MULT="$ATR_STOP_MULT" \
freqtrade backtesting \
  --config "$TEST_CONFIG" \
  --strategy RegimeMomentumV6 \
  --strategy-path "$STRATEGY_PATH" \
  --timerange "$TIMERANGE" \
  --timeframe-detail 5m \
  --fee 0.0004 \
  --breakdown month year
