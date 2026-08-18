#!/usr/bin/env bash
set -euo pipefail

START="${1:-2022-01-01}"
END="${2:-$(date -u -d 'yesterday' +%F)}"
LIVE_CONFIG="${RMV5_CONFIG:-/freqtrade/user_data/config-v5.generated.json}"
TEST_CONFIG="${ADAPTIVE_TEST_CONFIG:-/freqtrade/user_data/v5/config-adaptive20x-backtest.json}"
STRATEGY_PATH="${RMV5_STRATEGY_PATH:-/opt/rmv5/strategies}"
WALLET="${ADAPTIVE_BACKTEST_WALLET:-100}"
MAX_TRADES="${ADAPTIVE_MAX_OPEN_TRADES:-10}"

START_FT="${START//-/}"
END_FT="${END//-/}"
TIMERANGE="${START_FT}-${END_FT}"

python - "$LIVE_CONFIG" "$TEST_CONFIG" "$WALLET" "$MAX_TRADES" <<'PY'
import json, sys
src, dst, wallet, max_trades = sys.argv[1:]
cfg = json.load(open(src))
cfg["timeframe"] = "6h"
cfg["stake_amount"] = "unlimited"
cfg["tradable_balance_ratio"] = 0.99
cfg["dry_run_wallet"] = float(wallet)
cfg["max_open_trades"] = int(max_trades)
with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
print("AdaptiveTrend20x universe:", ", ".join(cfg["exchange"].get("pair_whitelist", [])))
print(f"wallet={wallet} USDT | max_open_trades={max_trades} | H6 | leverage=20x")
print("stake sizing: ~7% wallet per long, ~3% wallet per short")
PY

ADAPTIVE_FORCE_20X=true \
freqtrade backtesting \
  --config "$TEST_CONFIG" \
  --strategy AdaptiveTrend20x \
  --strategy-path "$STRATEGY_PATH" \
  --timerange "$TIMERANGE" \
  --timeframe-detail 5m \
  --breakdown month year
