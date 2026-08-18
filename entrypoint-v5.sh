#!/usr/bin/env bash
set -euo pipefail
mkdir -p /freqtrade/user_data/v5
TOP_N="${RMV5_TOP_N:-20}"

python /opt/rmv5/tools/select_universe.py \
  --top "$TOP_N" \
  --base-config /opt/rmv5/config-v5.base.json \
  --output-config /freqtrade/user_data/config-v5.generated.json \
  --universe /freqtrade/user_data/v5/universe.json

python /opt/rmv5/tools/marketdata_collector.py &
COLLECTOR_PID=$!

cleanup() {
  kill "$COLLECTOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

exec freqtrade trade \
  --config /freqtrade/user_data/config-v5.generated.json \
  --strategy RegimeMomentumV5 \
  --strategy-path /opt/rmv5/strategies \
  --db-url sqlite:////freqtrade/user_data/trades-rmv5.sqlite
