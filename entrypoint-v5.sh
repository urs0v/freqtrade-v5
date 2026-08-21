#!/usr/bin/env bash
set -euo pipefail

MODE="${RMV5_BOT_MODE:-frozen_fakeout}"

if [[ "$MODE" == "frozen_fakeout" ]]; then
  mkdir -p \
    /freqtrade/user_data/frozen_fakeout_feed \
    /freqtrade/user_data/frozen_fakeout_ws_detector \
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

  # Primary live signal path. It performs one concurrent REST cache catch-up,
  # bootstraps all 20 pairs against the frozen V1.6 causal reference, then moves
  # exclusively on Binance USD-M websocket 5m/15m events. Only parity-passed,
  # current and still-executable signals are published to Freqtrade.
  echo "FrozenFakeout deployment: starting websocket stateful execution feed."
  PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/frozen_fakeout_ws_execution.py \
    --outdir /freqtrade/user_data/frozen_fakeout_ws_detector \
    --feed-cache /freqtrade/user_data/frozen_fakeout_feed &
  WS_ENGINE_PID=$!

  export FROZEN_FAKEOUT_FEED="/freqtrade/user_data/frozen_fakeout_ws_detector/signals.csv"
  echo "FrozenFakeout deployment: starting Freqtrade dry-run API with websocket feed."
  freqtrade trade \
    --config /opt/rmv5/config-frozen-fakeout.dryrun.json \
    --strategy FrozenFakeoutV1 \
    --strategy-path /opt/rmv5/strategies \
    --db-url sqlite:////freqtrade/user_data/trades-frozen-fakeout.sqlite &
  FREQTRADE_PID=$!

  cleanup() {
    kill "$WS_ENGINE_PID" "$FREQTRADE_PID" 2>/dev/null || true
    wait "$WS_ENGINE_PID" "$FREQTRADE_PID" 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM

  # The bot must never appear healthy with a dead signal engine. If either the
  # websocket engine or Freqtrade exits, terminate the container so Docker/Coolify
  # can restart the complete pair together.
  set +e
  wait -n "$WS_ENGINE_PID" "$FREQTRADE_PID"
  STATUS=$?
  set -e
  echo "FrozenFakeout deployment: critical process exited status=$STATUS; restarting container." >&2
  exit "$STATUS"
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
