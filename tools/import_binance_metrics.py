"""
Import already-downloaded Binance Futures public metrics CSV/ZIP files into RMV5 features.sqlite.
For automatic public-data downloading use backfill_free.py.
"""
from __future__ import annotations
import argparse, sqlite3, zipfile
from pathlib import Path
import pandas as pd


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            if not names:
                raise ValueError(f"No CSV in {path}")
            with z.open(names[0]) as f:
                return pd.read_csv(f)
    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def to_timestamp(s: pd.Series) -> pd.Series:
    x = pd.to_datetime(s, utc=True, errors="coerce")
    if x.isna().mean() > 0.5:
        num = pd.to_numeric(s, errors="coerce")
        med = num.dropna().median()
        unit = "us" if med > 1e14 else "ms"
        x = pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
    return x


def ensure_db(db: Path):
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS features (
            bucket_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            oi REAL,
            funding_rate REAL,
            long_liq_usdt REAL,
            short_liq_usdt REAL,
            taker_ratio REAL,
            top_ls_ratio REAL,
            liq_observed INTEGER NOT NULL DEFAULT 0,
            updated_ms INTEGER NOT NULL,
            PRIMARY KEY (bucket_ms, symbol)
        )
    """)
    cols = {r[1] for r in con.execute("PRAGMA table_info(features)")}
    if "liq_observed" not in cols:
        con.execute("ALTER TABLE features ADD COLUMN liq_observed INTEGER NOT NULL DEFAULT 0")
    con.commit()
    return con


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--db", default="/freqtrade/user_data/v5/features.sqlite")
    args = ap.parse_args()

    con = ensure_db(Path(args.db))
    rows = 0
    for fname in args.files:
        p = Path(fname)
        df = read_any(p)
        if df.empty:
            continue
        tcol = "create_time" if "create_time" in df.columns else "timestamp"
        if tcol not in df.columns or "symbol" not in df.columns:
            print("skip unsupported:", p)
            continue
        df["date"] = to_timestamp(df[tcol])
        df = df.dropna(subset=["date"])
        df["bucket"] = df["date"].dt.floor("15min")
        oi_col = "sum_open_interest" if "sum_open_interest" in df.columns else None
        taker_col = "sum_taker_long_short_vol_ratio" if "sum_taker_long_short_vol_ratio" in df.columns else None
        top_col = "sum_toptrader_long_short_ratio" if "sum_toptrader_long_short_ratio" in df.columns else None

        aggmap = {}
        if oi_col: aggmap[oi_col] = "last"
        if taker_col: aggmap[taker_col] = "last"
        if top_col: aggmap[top_col] = "last"
        if not aggmap:
            continue
        grouped = df.groupby(["symbol", "bucket"], as_index=False).agg(aggmap)

        for _, r in grouped.iterrows():
            b = int(r["bucket"].timestamp() * 1000)
            con.execute("""
                INSERT INTO features
                (bucket_ms, symbol, oi, funding_rate, long_liq_usdt, short_liq_usdt,
                 taker_ratio, top_ls_ratio, liq_observed, updated_ms)
                VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?, 0, ?)
                ON CONFLICT(bucket_ms, symbol) DO UPDATE SET
                  oi=COALESCE(excluded.oi, features.oi),
                  taker_ratio=COALESCE(excluded.taker_ratio, features.taker_ratio),
                  top_ls_ratio=COALESCE(excluded.top_ls_ratio, features.top_ls_ratio)
            """, (
                b, str(r["symbol"]).upper(),
                float(r[oi_col]) if oi_col and pd.notna(r[oi_col]) else None,
                float(r[taker_col]) if taker_col and pd.notna(r[taker_col]) else None,
                float(r[top_col]) if top_col and pd.notna(r[top_col]) else None,
                b,
            ))
            rows += 1
        con.commit()
        print("imported", p)

    con.close()
    print("rows:", rows)


if __name__ == "__main__":
    main()
