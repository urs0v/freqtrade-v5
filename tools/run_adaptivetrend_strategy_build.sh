#!/usr/bin/env bash
set -euo pipefail

ROOT="${AT_BUILD_ROOT:-/freqtrade/user_data/strategy_build/adaptivetrend}"
DB="${AT_BUILD_DB:-$ROOT/core.sqlite}"
LOGDIR="$ROOT/logs"
mkdir -p "$ROOT" "$LOGDIR"

WORKERS="${AT_BUILD_WORKERS:-12}"

echo "=== STRATEGY BUILD: ADAPTIVETREND DOCUMENTED CORE ==="
echo "Stage 1/3: historical Binance H6 + exact funding, broad archive universe"
for PASS in 1 2 3; do
  set +e
  python -u /opt/rmv5/tools/backfill_adaptivetrend_core_data.py \
    --db "$DB" \
    --start "2021-01-01" \
    --end "2024-12-31" \
    --workers "$WORKERS" \
    2>&1 | tee -a "$LOGDIR/01_binance_core.log"
  RC=${PIPESTATUS[0]}
  set -e
  if [[ "$RC" -eq 0 ]]; then
    break
  fi
  if [[ "$PASS" -eq 3 ]]; then
    echo "Binance backfill still has errors after 3 resume passes. Stop before research."
    exit "$RC"
  fi
  echo "Transient Binance errors remain; resume pass $((PASS+1))/3 in 3s..."
  sleep 3
done

echo
echo "Stage 2/3: CoinGecko historical market caps (resume-aware)"
python -u /opt/rmv5/tools/backfill_adaptivetrend_market_caps.py \
  --db "$DB" \
  --start "2020-12-01" \
  --end "2025-01-02" \
  2>&1 | tee "$LOGDIR/02_market_caps.log"

echo
echo "Stage 3/3: 1x documented-core portfolio replication, OOS 2022-2024"
python -u /opt/rmv5/tools/backtest_adaptivetrend_documented_core.py \
  --db "$DB" \
  --start "2022-01-01" \
  --end "2025-01-01" \
  --outdir "$ROOT/results" \
  2>&1 | tee "$LOGDIR/03_replication.log"

echo
echo "=== STRATEGY BUILD COMPLETE ==="
echo "Results: $ROOT/results"
