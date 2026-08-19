from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_BASE_PATH = Path(__file__).with_name("AdaptivePerp15mV7.py")
_SPEC = importlib.util.spec_from_file_location("_adaptive_perp_v7_core", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load V7 core strategy from {_BASE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_BaseAdaptivePerp15mV7 = _MODULE.AdaptivePerp15mV7


class AdaptivePerp15mV7R1(_BaseAdaptivePerp15mV7):
    """V7-Core R1.

    Keeps the alpha/entry/exit logic of the first V7-Core run unchanged and fixes
    the risk governor that could effectively self-lock after a deep drawdown.

    The governor now reduces risk continuously from a configurable drawdown
    threshold to a configurable floor. It never intentionally disables trading;
    exchange minimum stake and portfolio-heat limits remain the hard constraints.
    This preserves the first run as a reproducible baseline while allowing a true
    Jan-Aug evaluation of the same alpha core.
    """

    def _drawdown_multiplier(self, equity: float) -> float:
        if self._equity_hwm is None or not np.isfinite(self._equity_hwm):
            self._equity_hwm = max(float(equity), 1e-9)

        self._equity_hwm = max(float(self._equity_hwm), float(equity))
        dd = max(0.0, 1.0 - float(equity) / max(float(self._equity_hwm), 1e-9))

        soft_start = self._env_float("RMV7_DD_SOFT_START", 0.08)
        full_at = self._env_float("RMV7_DD_FULL_AT", 0.30)
        floor = self._env_float("RMV7_DD_RISK_FLOOR", 0.45)

        soft_start = float(np.clip(soft_start, 0.0, 0.95))
        full_at = float(np.clip(full_at, soft_start + 1e-6, 0.99))
        floor = float(np.clip(floor, 0.10, 1.00))

        if dd <= soft_start:
            return 1.0

        x = float(np.clip((dd - soft_start) / (full_at - soft_start), 0.0, 1.0))
        # Smoothstep avoids abrupt risk jumps around arbitrary DD boundaries.
        smooth = x * x * (3.0 - 2.0 * x)
        return float(1.0 - (1.0 - floor) * smooth)
