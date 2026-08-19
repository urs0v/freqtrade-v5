from __future__ import annotations

import importlib.util
from pathlib import Path
from datetime import datetime


_BASE_PATH = Path(__file__).with_name("AdaptivePerp15mV7R1.py")
_SPEC = importlib.util.spec_from_file_location("_adaptive_perp_v7_r1", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Cannot load V7R1 strategy from {_BASE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_BaseAdaptivePerp15mV7R1 = _MODULE.AdaptivePerp15mV7R1


class AdaptivePerp15mV7Audit(_BaseAdaptivePerp15mV7R1):
    """Diagnostic wrapper around V7R1.

    Entry/exit indicators and signal generation are unchanged. Risk/execution
    overlays are neutralized so we can study the alpha separately from leverage,
    drawdown scaling, portfolio heat and protections.
    """

    @property
    def protections(self):
        return []

    def _drawdown_multiplier(self, equity: float) -> float:
        return 1.0

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return 1.0

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        stake = min(float(proposed_stake), float(max_stake))
        if min_stake is not None and stake < float(min_stake):
            return 0.0
        return max(stake, 0.0)
