#!/bin/bash
set -euo pipefail

OUT="/freqtrade/user_data/funding_edge_v1"
mkdir -p "$OUT"

echo "=== FUNDING EDGE V1 ==="
echo "Existing Binance futures cache only; no downloads."
echo "Causal funding features -> post-funding 1m entry -> forward returns."
echo "Discovery: 2025-11-01..2026-03-31 | Validation: 2026-04-01..2026-05-31 | Holdout: 2026-06-01..2026-08-19"

python /opt/rmv5/tools/funding_edge_v1.py \
  --outdir "$OUT" \
  --fee-bps-side "${FEE_BPS_SIDE:-5}" \
  --slippage-bps-side "${SLIPPAGE_BPS_SIDE:-1}" \
  --rolling-events "${ROLLING_EVENTS:-90}" \
  --min-history "${MIN_HISTORY:-45}"

echo ""
echo "Summary: $OUT/summary.json"
echo "Stats: $OUT/horizon_stats.csv"
echo "Signals: $OUT/signal_events.csv"
