#!/usr/bin/env bash
set -euo pipefail
V47DIR="${V47DIR:-/freqtrade/user_data/digash_fidelity_v47}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/digash_fidelity_v48}"
WORKERS="${WORKERS:-4}"
PRE_MINUTES="${PRE_MINUTES:-40}"
POST_MINUTES="${POST_MINUTES:-5}"
mkdir -p "$OUTDIR"
echo "=== DIGASH FIDELITY V4.8 RUNNER ==="
echo "AGGTRADES MICROSTRUCTURE: narrow public Binance Futures trade windows around exact source-level crosses."
echo "This run DOES download aggTrades, but only the required cross windows; cache is reused on rerun."
echo "WORKERS=$WORKERS PRE_MINUTES=$PRE_MINUTES POST_MINUTES=$POST_MINUTES"
echo "V47DIR=$V47DIR OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/audit_digash_fidelity_v48.py \
  --v47dir "$V47DIR" \
  --outdir "$OUTDIR" \
  --workers "$WORKERS" \
  --pre-minutes "$PRE_MINUTES" \
  --post-minutes "$POST_MINUTES" \
  "$@" 2>&1 | tee "$OUTDIR/run.log"
