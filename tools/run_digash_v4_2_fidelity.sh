#!/bin/bash
set -euo pipefail

OUT="/freqtrade/user_data/digash_v4_2_fidelity"
SIM="$OUT/sim"
mkdir -p "$OUT" "$SIM"

echo "=== DIGASH V4.2 QUALITY-FIRST FIDELITY + PNL REPLAY ==="
echo "Detector first; post-signal replay second. Frequency is NOT an objective."
echo "Output: $OUT"

python /opt/rmv5/tools/digash_v4_2_fidelity.py \
  --outdir "$OUT" \
  --gold /opt/rmv5/tools/digash_v4_2_gold.csv \
  --start 2025-11-01 \
  --end 2026-08-19 \
  --workers "${WORKERS:-12}" \
  --sample 100

echo ""
echo "=== STARTING POST-SIGNAL PNL REPLAY ==="
python /opt/rmv5/tools/digash_v4_2_sim.py \
  --events "$OUT/events.csv" \
  --outdir "$SIM" \
  --starting-equity "${STARTING_EQUITY:-100}" \
  --risk-pcts "${RISK_PCTS:-0.01,0.02,0.03}" \
  --max-leverage "${MAX_LEVERAGE:-10}" \
  --max-concurrent "${MAX_CONCURRENT:-3}" \
  --fee-bps-side "${FEE_BPS_SIDE:-5}" \
  --slippage-bps-side "${SLIPPAGE_BPS_SIDE:-1}" \
  --max-hold-hours "${MAX_HOLD_HOURS:-24}"

echo ""
echo "Fidelity summary: $OUT/summary.json"
echo "PnL replay summary: $SIM/summary.json"
