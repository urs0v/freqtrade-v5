#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd
import research_derivatives_alpha as r

_original_load_derivatives = r.load_derivatives


def load_derivatives_ns(db, symbol, start_ms, end_ms):
    df = _original_load_derivatives(db, symbol, start_ms, end_ms)
    if not df.empty:
        df = df.copy()
        df["available_time"] = pd.to_datetime(df["available_time"], utc=True).astype("datetime64[ns, UTC]")
    return df


r.load_derivatives = load_derivatives_ns

if __name__ == "__main__":
    raise SystemExit(r.main())
