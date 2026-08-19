#!/usr/bin/env bash
set -euo pipefail

# User-facing dates are inclusive. Freqtrade's timerange end is exclusive.
START="${1:-2026-01-01}"
END_INCLUSIVE="${2:-2026-08-18}"

LIVE_CONFIG="${RMV5_CONFIG:-/freqtrade/user_data/config-v5.generated.json}"
BASE_CONFIG="${RMV7_BASE_CONFIG:-/opt/rmv5/config-v5.base.json}"
TEST_CONFIG="${RMV7_TEST_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
UNIVERSE_FILE="${RMV7_UNIVERSE_FILE:-/opt/rmv5/tools/historical_universe.json}"
STRATEGY_PATH="${RMV5_STRATEGY_PATH:-/opt/rmv5/strategies}"

WALLET="${RMV7_BACKTEST_WALLET:-100}"
MAX_TRADES="${RMV7_MAX_OPEN_TRADES:-5}"
FEE="${RMV7_BACKTEST_FEE:-0.0004}"

ENTRY_THRESHOLD="${RMV7_ENTRY_THRESHOLD:-0.64}"
SCORE_GAP="${RMV7_SCORE_GAP:-0.08}"
LEVERAGE_MIN="${RMV7_LEVERAGE_MIN:-3}"
LEVERAGE_MAX="${RMV7_LEVERAGE_MAX:-10}"
RISK_MIN="${RMV7_RISK_MIN:-0.0075}"
RISK_MAX="${RMV7_RISK_MAX:-0.0200}"
PORTFOLIO_HEAT="${RMV7_PORTFOLIO_HEAT:-0.08}"
SIDE_HEAT="${RMV7_SIDE_HEAT:-0.05}"
ATR_STOP_MULT="${RMV7_ATR_STOP_MULT:-1.8}"

END_EXCLUSIVE="$(python - "$END_INCLUSIVE" <<'PY'
import sys
from datetime import datetime, timedelta
x = datetime.strptime(sys.argv[1], '%Y-%m-%d') + timedelta(days=1)
print(x.strftime('%Y-%m-%d'))
PY
)"
WARMUP_START="$(python - "$START" <<'PY'
import sys
from datetime import datetime, timedelta
x = datetime.strptime(sys.argv[1], '%Y-%m-%d') - timedelta(days=90)
print(x.strftime('%Y-%m-%d'))
PY
)"

START_FT="${START//-/}"
END_EXCLUSIVE_FT="${END_EXCLUSIVE//-/}"
WARMUP_FT="${WARMUP_START//-/}"
TIMERANGE="${START_FT}-${END_EXCLUSIVE_FT}"
DOWNLOAD_RANGE="${WARMUP_FT}-${END_EXCLUSIVE_FT}"

mkdir -p /freqtrade/user_data/v7

SOURCE_CONFIG="$LIVE_CONFIG"
if [ ! -f "$SOURCE_CONFIG" ]; then
  SOURCE_CONFIG="$BASE_CONFIG"
fi

echo "=== 1/4 BUILD REPRODUCIBLE V7 CORE CONFIG ==="
python - "$SOURCE_CONFIG" "$UNIVERSE_FILE" "$TEST_CONFIG" "$WALLET" "$MAX_TRADES" <<'PY'
import json, sys
from pathlib import Path

source, universe_file, output, wallet, max_trades = sys.argv[1:]
cfg = json.loads(Path(source).read_text())

u = Path(universe_file)
if u.exists():
    uni = json.loads(u.read_text())
    symbols = [str(x).upper() for x in uni.get("symbols", [])]
    pairs = [f"{s[:-4]}/USDT:USDT" for s in symbols if s.endswith("USDT")]
    if pairs:
        cfg["exchange"]["pair_whitelist"] = pairs

cfg["timeframe"] = "15m"
cfg["stake_amount"] = "unlimited"
cfg["tradable_balance_ratio"] = 0.99
cfg["dry_run"] = True
cfg["dry_run_wallet"] = float(wallet)
cfg["max_open_trades"] = int(max_trades)
cfg["trading_mode"] = "futures"
cfg["margin_mode"] = "isolated"
cfg["liquidation_buffer"] = max(float(cfg.get("liquidation_buffer", 0.12)), 0.12)
cfg["bot_name"] = "AdaptivePerp15mV7-Core"

# Market execution avoids optimistic historical limit-fill assumptions in the core test.
order_types = cfg.setdefault("order_types", {})
order_types["entry"] = "market"
order_types["exit"] = "market"
order_types["emergency_exit"] = "market"
order_types["force_entry"] = "market"
order_types["force_exit"] = "market"
order_types["stoploss"] = "market"
order_types["stoploss_on_exchange"] = False

# API server is irrelevant for a research backtest and should not pull secrets into output.
if "api_server" in cfg:
    cfg["api_server"]["enabled"] = False

out = Path(output)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(cfg, indent=2))
print(f"Config: {out}")
print(f"Universe ({len(cfg['exchange'].get('pair_whitelist', []))}): " + ", ".join(cfg['exchange'].get('pair_whitelist', [])))
PY

echo "=== 2/4 ENSURE 1M / 15M / 1H / 4H FUTURES DATA ==="
echo "Warmup: $WARMUP_START | research: $START -> $END_INCLUSIVE"
freqtrade download-data \
  --config "$TEST_CONFIG" \
  --trading-mode futures \
  --timeframes 1m 15m 1h 4h \
  --timerange "$DOWNLOAD_RANGE"

echo "=== 3/4 ENSURE 1H FUNDING + MARK DATA ==="
# Current Freqtrade represents futures funding/mark candles on 1h. If the exchange
# already supplied them, this simply reuses/extends the local dataset.
freqtrade download-data \
  --config "$TEST_CONFIG" \
  --trading-mode futures \
  --timeframes 1h \
  --candle-types funding_rate mark \
  --timerange "$DOWNLOAD_RANGE"

echo "=== 4/4 ADAPTIVEPERP15M V7 CORE BACKTEST ==="
echo "period=$START..$END_INCLUSIVE | wallet=$WALLET | max_open=$MAX_TRADES | fee=$FEE"
echo "score threshold=$ENTRY_THRESHOLD | gap=$SCORE_GAP"
echo "leverage=${LEVERAGE_MIN}-${LEVERAGE_MAX}x | risk/trade=${RISK_MIN}-${RISK_MAX} | heat=$PORTFOLIO_HEAT | side_heat=$SIDE_HEAT"
echo "ATR stop mult=$ATR_STOP_MULT | timeframe=15m | detail=1m"

RMV7_ENTRY_THRESHOLD="$ENTRY_THRESHOLD" \
RMV7_SCORE_GAP="$SCORE_GAP" \
RMV7_LEVERAGE_MIN="$LEVERAGE_MIN" \
RMV7_LEVERAGE_MAX="$LEVERAGE_MAX" \
RMV7_RISK_MIN="$RISK_MIN" \
RMV7_RISK_MAX="$RISK_MAX" \
RMV7_PORTFOLIO_HEAT="$PORTFOLIO_HEAT" \
RMV7_SIDE_HEAT="$SIDE_HEAT" \
RMV7_ATR_STOP_MULT="$ATR_STOP_MULT" \
RMV7_ENABLE_PROTECTIONS="${RMV7_ENABLE_PROTECTIONS:-true}" \
RMV7_ENABLE_TIME_EXIT="${RMV7_ENABLE_TIME_EXIT:-true}" \
freqtrade backtesting \
  --config "$TEST_CONFIG" \
  --strategy AdaptivePerp15mV7 \
  --strategy-path "$STRATEGY_PATH" \
  --timerange "$TIMERANGE" \
  --timeframe 15m \
  --timeframe-detail 1m \
  --fee "$FEE" \
  --enable-protections \
  --cache none \
  --export trades \
  --breakdown month year
