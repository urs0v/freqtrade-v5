#!/usr/bin/env bash
set -euo pipefail

START="${START:-2026-05-01}"
END="${END:-2026-08-18}"
ROOT="${ROOT:-/freqtrade/user_data/qh_edge}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-30}"

python - <<'PY'
from datetime import datetime, timedelta
from pathlib import Path
import os
start=datetime.strptime(os.environ.get('START','2026-05-01'),'%Y-%m-%d').date()
end=datetime.strptime(os.environ.get('END','2026-08-18'),'%Y-%m-%d').date()
root=Path(os.environ.get('ROOT','/freqtrade/user_data/qh_edge'))
missing=[]
d=start
while d<=end:
    p=root/'cache'/'BTCUSDT'/f'{d.isoformat()}.npz'
    if not p.exists(): missing.append(str(d))
    d += timedelta(days=1)
if missing:
    raise SystemExit('CACHE_MISSING: '+','.join(missing))
print(f'CACHE_OK: all BTC days present for {start}..{end}')
PY

echo "=== BTC CAUSAL QH REGIME AUDIT ==="
echo "Cache-only. No downloads."
echo "START=$START END=$END LOOKBACK_DAYS=$LOOKBACK_DAYS ROOT=$ROOT"
echo

python /opt/rmv5/tools/qh_causal_regime_cached.py \
  --start "$START" \
  --end "$END" \
  --symbol BTCUSDT \
  --root "$ROOT" \
  --lookback-days "$LOOKBACK_DAYS" \
  2>&1 | tee "$ROOT/causal_regime_btc.log"
