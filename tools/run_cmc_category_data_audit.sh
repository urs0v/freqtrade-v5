#!/usr/bin/env bash
set -euo pipefail

OUT="/freqtrade/user_data/cmc_category_audit"
VENDOR="$OUT/vendor"
mkdir -p "$OUT" "$VENDOR"

if ! PYTHONPATH="$VENDOR" python - <<'PY' >/dev/null 2>&1
import kagglehub
PY
then
  echo "Installing kagglehub into research-only user_data directory..."
  python -m pip install --quiet --disable-pip-version-check --target "$VENDOR" "kagglehub>=1.0,<2"
fi

echo "=== CMC CATEGORY DATA AUDIT RUNNER ==="
echo "This is NOT a strategy backtest."
echo "It downloads only coins.csv metadata from a small sample of historical Kaggle dataset versions."
echo

PYTHONPATH="$VENDOR" python /opt/rmv5/tools/audit_cmc_category_versions.py --out "$OUT" 2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Paste the entire block from '=== CMC CATEGORY HISTORY DATA-VIABILITY AUDIT ==='."
