#!/usr/bin/env bash
set -euo pipefail

START_DATE="${1:-2021-08-20}"
END_DATE="${2:-2026-08-19}"

START_COMPACT="${START_DATE//-/}"
END_COMPACT="${END_DATE//-/}"
TIMERANGE="${START_COMPACT}-${END_COMPACT}"

CONFIG="/opt/rmv5/config-v5.base.json"
STRATEGY_PATH="/opt/rmv5/strategies"
STRATEGY="VolumeProfileMeanReversionV0"
PAIR="SOL/USDT:USDT"
OUT="/freqtrade/user_data/vpmr_v0"

mkdir -p "$OUT"

echo "=== VPMR V0: REAL BINANCE SOL/USDT PERPETUAL BACKTEST ==="
echo "Range: $START_DATE -> $END_DATE"
echo "Strategy: paper rules, developing POC (no full-day lookahead), 1x leverage"
echo "Cost stress: 0.07% per side = 0.14% round trip"
echo

echo "[1/4] Downloading real Binance USD-M futures candles (1m + 5m)..."
freqtrade download-data \
  -c "$CONFIG" \
  --trading-mode futures \
  --pairs "$PAIR" \
  --timerange "$TIMERANGE" \
  --timeframes 1m 5m \
  --candle-types futures

echo
echo "[2/4] Downloading funding-rate and mark data..."
freqtrade download-data \
  -c "$CONFIG" \
  --trading-mode futures \
  --pairs "$PAIR" \
  --timerange "$TIMERANGE" \
  --timeframes 1h \
  --candle-types funding_rate mark

echo
echo "[3/4] Verifying downloaded candle coverage..."
freqtrade list-data \
  -c "$CONFIG" \
  --trading-mode futures \
  --pairs "$PAIR" \
  --show-timerange | tee "$OUT/data_coverage.log"

echo
echo "[4/4] Running stop-loss sensitivity: 2%, 3%, 5%..."
for SL in 0.02 0.03 0.05; do
  export VPMR_STOPLOSS="$SL"
  LABEL="${SL/0./}pct"
  LOG="$OUT/vpmr_v0_sl_${LABEL}.log"
  RESULT_DIR="$OUT/sl_${LABEL}"
  mkdir -p "$RESULT_DIR"

  echo
  echo "------------------------------------------------------------"
  echo "STOP LOSS = $SL"
  echo "Full output: $LOG"
  echo "------------------------------------------------------------"

  freqtrade backtesting \
    -c "$CONFIG" \
    -s "$STRATEGY" \
    --strategy-path "$STRATEGY_PATH" \
    --pairs "$PAIR" \
    --timerange "$TIMERANGE" \
    --timeframe 5m \
    --timeframe-detail 1m \
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
  LOG="$OUT/vpmr_v0_sl_${LABEL}.log"
  echo
  echo "--- STOP LOSS $SL ---"
  grep -E \
    "Backtesting from|Backtesting to|Total/Daily Avg Trades|Starting balance|Final balance|Total profit %|Profit factor|Expectancy \(Ratio\)|Long / Short trades|Long / Short profit %|Max % of account underwater|Absolute drawdown|Sharpe \(closed trades\)|Max Consecutive Wins / Loss" \
    "$LOG" || true
  echo "Strategy summary:"
  awk '/STRATEGY SUMMARY/{flag=1; count=0} flag{print; count++} flag && count>=12{flag=0}' "$LOG" | tail -n 12 || true
done

echo
echo "=== DATA COVERAGE ==="
cat "$OUT/data_coverage.log" || true

echo
echo "=== DONE ==="
echo "All logs/results are under: $OUT"
echo "Paste everything from '=== COMPACT SUMMARY ===' through '=== DONE ===' into ChatGPT."
