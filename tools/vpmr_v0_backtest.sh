#!/usr/bin/env bash
set -euo pipefail

START_DATE="${1:-2021-08-20}"
END_DATE="${2:-2026-08-19}"

START_COMPACT="${START_DATE//-/}"
END_COMPACT="${END_DATE//-/}"
TIMERANGE="${START_COMPACT}-${END_COMPACT}"

BASE_CONFIG="/opt/rmv5/config-v5.base.json"
STRATEGY_PATH="/opt/rmv5/strategies"
STRATEGY="VolumeProfileMeanReversionV0"
PAIR="SOL/USDT:USDT"
OUT="/freqtrade/user_data/vpmr_v0"
TEST_CONFIG="$OUT/config-vpmr-v0-backtest.json"

mkdir -p "$OUT"

# The production config uses stake_amount=50 with a $100 wallet. A losing research
# strategy then stops trading as soon as equity falls below $50, which truncates
# the sample and makes later years disappear. For alpha testing, use a small fixed
# $10 stake so all years can be observed while preserving the requested $100 wallet.
python - "$BASE_CONFIG" "$TEST_CONFIG" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg["stake_amount"] = 10
cfg["max_open_trades"] = 1
cfg["dry_run_wallet"] = 100
cfg["pairlists"] = [{"method": "StaticPairList"}]
cfg.setdefault("exchange", {})["pair_whitelist"] = ["SOL/USDT:USDT"]
cfg["exchange"]["pair_blacklist"] = []

with open(dst, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
PY

echo "=== VPMR V0: REAL BINANCE SOL/USDT PERPETUAL BACKTEST ==="
echo "Range: $START_DATE -> $END_DATE"
echo "Strategy: paper rules, developing POC (no full-day lookahead), 1x leverage"
echo "Wallet: 100 USDT | fixed research stake: 10 USDT"
echo "Cost stress: 0.07% per side = 0.14% round trip"
echo "Execution resolution: 5m only (matches the paper; no partial 1m mixing)"
echo

echo "[1/3] Ensuring real Binance USD-M 5m futures candles are present..."
freqtrade download-data \
  -c "$TEST_CONFIG" \
  --trading-mode futures \
  --pairs "$PAIR" \
  --timerange "$TIMERANGE" \
  --timeframes 5m \
  --candle-types futures

echo
echo "[2/3] Verifying downloaded candle coverage..."
freqtrade list-data \
  -c "$TEST_CONFIG" \
  --trading-mode futures \
  --pairs "$PAIR" \
  --show-timerange | tee "$OUT/data_coverage.log"

echo
echo "[3/3] Running stop-loss sensitivity: 2%, 3%, 5%..."
for SL in 0.02 0.03 0.05; do
  export VPMR_STOPLOSS="$SL"
  LABEL="${SL/0./}pct"
  LOG="$OUT/vpmr_v0_full_sl_${LABEL}.log"
  RESULT_DIR="$OUT/full_sl_${LABEL}"
  mkdir -p "$RESULT_DIR"

  echo
  echo "------------------------------------------------------------"
  echo "STOP LOSS = $SL"
  echo "Full output: $LOG"
  echo "------------------------------------------------------------"

  freqtrade backtesting \
    -c "$TEST_CONFIG" \
    -s "$STRATEGY" \
    --strategy-path "$STRATEGY_PATH" \
    --pairs "$PAIR" \
    --timerange "$TIMERANGE" \
    --timeframe 5m \
    --starting-balance 100 \
    --max-open-trades 1 \
    --fee 0.0007 \
    --cache none \
    --breakdown month year \
    --export trades \
    --backtest-directory "$RESULT_DIR" \
    2>&1 | tee "$LOG"
done

echo
echo "=== COMPACT SUMMARY ==="
for SL in 0.02 0.03 0.05; do
  LABEL="${SL/0./}pct"
  LOG="$OUT/vpmr_v0_full_sl_${LABEL}.log"
  echo
  echo "--- STOP LOSS $SL ---"
  grep -E \
    "Backtesting from|Backtesting to|Total/Daily Avg Trades|Starting balance|Final balance|Total profit %|Profit factor|Expectancy \(Ratio\)|Long / Short trades|Long / Short profit %|Max % of account underwater|Absolute drawdown|Sharpe \(closed trades\)|Max Consecutive Wins / Loss|Best trade|Worst trade" \
    "$LOG" || true
  echo "Strategy summary:"
  awk '/STRATEGY SUMMARY/{flag=1; count=0} flag{print; count++} flag && count>=12{flag=0}' "$LOG" | tail -n 12 || true
  echo "Year breakdown:"
  awk '/YEAR BREAKDOWN/{flag=1; count=0} flag{print; count++} flag && count>=14{flag=0}' "$LOG" | tail -n 14 || true
done

echo
echo "=== DATA COVERAGE ==="
cat "$OUT/data_coverage.log" || true

echo
echo "=== DONE ==="
echo "All logs/results are under: $OUT"
echo "Paste everything from '=== COMPACT SUMMARY ===' through '=== DONE ===' into ChatGPT."
