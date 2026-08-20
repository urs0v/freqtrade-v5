#!/usr/bin/env bash
set -euo pipefail

HOURS="${1:-6}"
OUT="/freqtrade/user_data/mm_shadow_btc"
mkdir -p "$OUT"

python -m py_compile /opt/rmv5/tools/mm_shadow_btc.py /opt/rmv5/tools/mm_shadow_report.py
rm -f "$OUT/mm_shadow.sqlite"

SECS=$(python - <<PY
print(float("$HOURS") * 3600.0)
PY
)

echo "=== BTCUSDT LIVE SHADOW MARKET MAKER ==="
echo "Runtime: ${HOURS}h"
echo "Output:  $OUT"
echo "Fresh experiment DB created for this run."
echo "SHADOW ONLY: no API keys and no real orders."
echo "Public Binance Futures WS: aggTrade + diff depth + bookTicker + markPrice."
echo "Conservative fill model: our virtual order starts behind displayed quantity at its price; only actual aggressive trades consume queue."
echo

python /opt/rmv5/tools/mm_shadow_btc.py \
  --symbol BTCUSDT \
  --output-dir "$OUT" \
  --runtime-seconds "$SECS" \
  --virtual-capital 100 \
  --quote-notional 10 \
  --max-inventory-notional 30 \
  --maker-fee-bps 2.0 \
  --min-halfspread-bps 2.5 \
  --vol-mult 0.75 \
  --vol-gate-mult 2.5 \
  --inventory-skew-bps 2.0 \
  --quote-refresh-ms 1000 \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== FINAL REPORT ==="
python /opt/rmv5/tools/mm_shadow_report.py --db "$OUT/mm_shadow.sqlite"
