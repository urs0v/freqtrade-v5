#!/usr/bin/env bash
set -euo pipefail

CONFIG="${DERIV_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DB="${DERIV_DB:-/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite}"
CACHE="${DERIV_CACHE:-/freqtrade/user_data/v5/free-cache}"
START="${DERIV_START:-2022-01-01}"
END="${DERIV_END:-2026-08-19}"
CONCURRENCY="${DERIV_CONCURRENCY:-24}"

if [ ! -f "$CONFIG" ]; then
  echo "Missing config: $CONFIG"
  exit 2
fi

mkdir -p "$(dirname "$DB")" "$CACHE"

SYMBOLS=$(python - "$CONFIG" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
pairs = cfg.get("exchange", {}).get("pair_whitelist", [])
out = []
for p in pairs:
    base = p.split("/")[0]
    out.append(base + "USDT")
print(",".join(dict.fromkeys(out)))
PY
)

if [ -z "$SYMBOLS" ]; then
  echo "No pair_whitelist in $CONFIG"
  exit 2
fi

echo "=== DERIVATIVES ALPHA LAB: POINT-IN-TIME BACKFILL ==="
echo "DB: $DB"
echo "Range: $START -> $END"
echo "Symbols: $SYMBOLS"
echo "Historical metrics are stamped at 15m bucket CLOSE; funding is strictly lagged past its event timestamp."
echo "Starting downloader (progress prints every 250 jobs)..."

# -u keeps progress visible even when stdout is piped through tee/Coolify.
python -u /opt/rmv5/tools/backfill_derivatives_pti.py \
  --start "$START" \
  --end "$END" \
  --symbols "$SYMBOLS" \
  --db "$DB" \
  --cache "$CACHE" \
  --concurrency "$CONCURRENCY"

echo "Backfill finished: $DB"
