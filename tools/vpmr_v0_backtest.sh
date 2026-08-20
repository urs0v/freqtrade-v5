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

echo "[1/3] Downloading real Binance USD-M futures candles (1m + 5m)..."
freqtrade download-data \
  -c "$CONFIG" \
  --trading-mode futures \
  --pairs "$PAIR" \
  --timerange "$TIMERANGE" \
  --timeframes 1m 5m \
  --candle-types futures

echo
echo "[2/3] Downloading funding-rate and mark data..."
freqtrade download-data \
  -c "$CONFIG" \
  --trading-mode futures \
  --pairs "$PAIR" \
  --timerange "$TIMERANGE" \
  --timeframes 1h \
  --candle-types funding_rate mark

echo
echo "[3/3] Running stop-loss sensitivity: 2%, 3%, 5%..."
for SL in 0.02 0.03 0.05; do
  export VPMR_STOPLOSS="$SL"
  LABEL="${SL/0./}pct"
  RESULT="$OUT/vpmr_v0_sl_${LABEL}.json"

  echo
  echo "------------------------------------------------------------"
  echo "STOP LOSS = $SL"
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
    --export-filename "$RESULT"
done

echo
echo "=== DONE ==="
echo "Results are under: $OUT"
echo "Paste the three Freqtrade summary tables back into ChatGPT."
