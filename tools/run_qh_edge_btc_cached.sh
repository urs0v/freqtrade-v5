#!/usr/bin/env bash
set -euo pipefail

START="${START:-2026-05-01}"
END="${END:-2026-08-18}"
ROOT="${ROOT:-/freqtrade/user_data/qh_edge}"
SYMBOL="BTCUSDT"

CACHE="$ROOT/cache/$SYMBOL"
REPORT="$ROOT/reports/${START}_${END}"

python - "$START" "$END" "$CACHE" <<'PY'
import sys
from datetime import datetime, timedelta
from pathlib import Path

start = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
end = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()
cache = Path(sys.argv[3])

missing = []
d = start
while d <= end:
    p = cache / f"{d.isoformat()}.npz"
    if not p.exists():
        missing.append(str(d))
    d += timedelta(days=1)

if missing:
    print(f"CACHE_INCOMPLETE: missing {len(missing)} BTC days")
    print("First missing days:", ", ".join(missing[:20]))
    print("No download was started.")
    raise SystemExit(2)

print(f"CACHE_OK: all BTC days present for {start}..{end}")
PY

echo "=== BTC QUARTER-HOUR CACHE-ONLY AUDIT ==="
echo "No network download should occur. Existing 10-second cache only."
echo "START=$START END=$END ROOT=$ROOT"
echo

python /opt/rmv5/tools/qh_edge_audit.py \
  --start "$START" \
  --end "$END" \
  --root "$ROOT" \
  --symbols "$SYMBOL" \
  2>&1 | tee "$ROOT/btc_cached_run.log"

echo
echo "=== BTC EXTREME FLOW CELLS ==="
if [[ -f "$REPORT/extreme_flow.csv" ]]; then
  cat "$REPORT/extreme_flow.csv"
fi

echo
echo "=== IMPORTANT ==="
echo "Ignore paper_edge_replicated/trading_candidate booleans in summary.json for this single-asset diagnostic:"
echo "the core script's final portfolio gates were pre-registered for the six-asset replication."
echo "For this run, paste the BTC 4h/8h/12h beta+t lines and BTC EXTREME FLOW CELLS."
