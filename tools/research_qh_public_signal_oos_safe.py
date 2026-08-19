#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import pandas as pd

import research_qh_public_signal_oos as m


def fit_asset_model_safe(df: pd.DataFrame, ti_cols: list[str]) -> dict:
    predictors = m.LAG_COLS + ti_cols
    train_end = pd.Timestamp(m.TRAIN_END, tz="UTC")
    train = df[
        (df["available_time"] >= pd.Timestamp(m.TRAIN_START, tz="UTC"))
        & (df["available_time"] < train_end)
        & (df[f"exit_time_{m.HORIZON}"] < train_end)
    ].copy()
    train = train.dropna(subset=["oi", f"y_{m.HORIZON}", *predictors])
    if len(train) < 1000:
        raise RuntimeError(f"Insufficient 2024 rows: {len(train)}")

    mu, sd = m.fit_standardizer(train, predictors)
    X = m.transform(train, predictors, mu, sd)
    y_oi = train["oi"].to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y_oi, rcond=None)
    intercept = float(beta[0])
    phi = beta[1:1 + len(m.LAG_COLS)]
    psi = beta[1 + len(m.LAG_COLS):]

    lag_comp = X[:, :len(m.LAG_COLS)] @ phi
    pub_comp = X[:, len(m.LAG_COLS):] @ psi
    residual = y_oi - lag_comp - pub_comp
    fitted = intercept + lag_comp + pub_comp
    denom = float(np.sum((y_oi - np.mean(y_oi)) ** 2))
    stage1_r2 = float(1.0 - np.sum((y_oi - fitted) ** 2) / denom) if denom > 0 else np.nan

    yret = train[f"y_{m.HORIZON}"].to_numpy(dtype=float)
    B = np.column_stack([np.ones(len(train)), lag_comp, pub_comp, residual])
    e, *_ = np.linalg.lstsq(B, yret, rcond=None)

    return {
        "predictors": predictors,
        "mu": mu,
        "sd": sd,
        "intercept_oi": intercept,
        "phi": phi,
        "psi": psi,
        "e0": float(e[0]),
        "e_lag": float(e[1]),
        "e_pub": float(e[2]),
        "e_res": float(e[3]),
        "stage1_r2": stage1_r2,
        "n_train_fit": int(len(train)),
    }


m.fit_asset_model = fit_asset_model_safe

if __name__ == "__main__":
    raise SystemExit(m.main())
