#!/usr/bin/env bash
set -euo pipefail

START="${1:-2022-01-01}"
END="${2:-$(date -u -d 'yesterday' +%F)}"
LIVE_CONFIG="${RMV5_CONFIG:-/freqtrade/user_data/config-v5.generated.json}"
TEST_CONFIG="${RMV5_TEST_CONFIG:-/freqtrade/user_data/v5/config-current-backtest.json}"
HIST_DB="${RMV5_HIST_DB:-/freqtrade/user_data/v5/features-backtest.sqlite}"
STRATEGY_PATH="${RMV5_STRATEGY_PATH:-/opt/rmv5/strategies}"
STAKE="${RMV5_BACKTEST_STAKE:-10}"
WALLET="${RMV5_BACKTEST_WALLET:-100}"

START_FT="${START//-/}"
END_FT="${END//-/}"
TIMERANGE="${START_FT}-${END_FT}"

python - "$LIVE_CONFIG" "$TEST_CONFIG" "$STAKE" "$WALLET" <<'PY'
import json, sys
src, dst, stake, wallet = sys.argv[1:]
cfg = json.load(open(src))
cfg["stake_amount"] = float(stake)
cfg["dry_run_wallet"] = float(wallet)
cfg["max_open_trades"] = 1
with open(dst, "w") as f:
    json.dump(cfg, f, indent=2)
print("Backtest universe:", ", ".join(cfg["exchange"].get("pair_whitelist", [])))
print(f"Backtest wallet={wallet} USDT, stake={stake} USDT, forced leverage=20x")
PY

RMV5_FEATURE_DB="$HIST_DB" RMV5_FORCE_20X=true \
freqtrade backtesting \
  --config "$TEST_CONFIG" \
  --strategy RegimeMomentumV5 \
  --strategy-path "$STRATEGY_PATH" \
  --timerange "$TIMERANGE" \
  --timeframe-detail 5m \
  --breakdown month year
