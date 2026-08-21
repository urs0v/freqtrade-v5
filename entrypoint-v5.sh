#!/usr/bin/env bash
set -euo pipefail

MODE="${RMV5_BOT_MODE:-frozen_fakeout}"

if [[ "$MODE" == "frozen_fakeout" ]]; then
  mkdir -p \
    /freqtrade/user_data/frozen_fakeout_feed \
    /freqtrade/user_data/frozen_fakeout_ws_shadow \
    /freqtrade/user_data/prospective_fakeout_v2

  PARITY_JSON="/freqtrade/user_data/prospective_fakeout_v2/parity_pass.json"
  if [[ ! -f "$PARITY_JSON" ]] || ! grep -q '"verdict": "PARITY_PASS"' "$PARITY_JSON"; then
    echo "FrozenFakeout deployment: historical parity gate is missing; running it now."
    PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/prospective_fakeout_v2_parity.py \
      --outdir /freqtrade/user_data/prospective_fakeout_v2
  fi
  if [[ ! -f "$PARITY_JSON" ]] || ! grep -q '"verdict": "PARITY_PASS"' "$PARITY_JSON"; then
    echo "FrozenFakeout deployment refused to start: PARITY_PASS is required." >&2
    exit 2
  fi

  echo "FrozenFakeout deployment: parity PASS; starting persistent incremental causal feed."
  PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/frozen_fakeout_signal_feed_v2.py \
    --loop \
    --outdir /freqtrade/user_data/frozen_fakeout_feed &
  FEED_PID=$!

  # Shadow-only transport probe. It never writes the execution feed and never
  # sends orders. This measures the real Binance USD-M websocket close/open
  # delivery latency before we move the frozen detector onto the websocket path.
  echo "FrozenFakeout deployment: starting Binance websocket shadow probe."
  PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/frozen_fakeout_ws_shadow.py \
    --outdir /freqtrade/user_data/frozen_fakeout_ws_shadow &
  WS_SHADOW_PID=$!

  cleanup() {
    kill "$FEED_PID" "$WS_SHADOW_PID" 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM

  export FROZEN_FAKEOUT_FEED="/freqtrade/user_data/frozen_fakeout_feed/signals.csv"
  echo "FrozenFakeout deployment: starting Freqtrade dry-run API on configured port."
  exec freqtrade trade \
    --config /opt/rmv5/config-frozen-fakeout.dryrun.json \
    --strategy FrozenFakeoutV1 \
    --strategy-path /opt/rmv5/strategies \
    --db-url sqlite:////freqtrade/user_data/trades-frozen-fakeout.sqlite
fi

if [[ "$MODE" != "regime_momentum_v5" ]]; then
  echo "Unknown RMV5_BOT_MODE=$MODE (expected frozen_fakeout or regime_momentum_v5)" >&2
  exit 2
fi

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
