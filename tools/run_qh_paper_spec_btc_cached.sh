#!/usr/bin/env bash
set -euo pipefail

START="${START:-2026-05-01}"
END="${END:-2026-08-18}"
ROOT="${ROOT:-/freqtrade/user_data/qh_edge}"

for d in $(seq -f '%02g' 1 1); do :; done

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

echo "=== BTC PAPER-SPEC CACHE-ONLY REPLICATION ==="
echo "No network download should occur."
echo "START=$START END=$END ROOT=$ROOT"
echo

python /opt/rmv5/tools/qh_paper_spec_cached.py \
  --start "$START" \
  --end "$END" \
  --symbol BTCUSDT \
  --root "$ROOT" \
  2>&1 | tee "$ROOT/paper_spec_btc.log"
