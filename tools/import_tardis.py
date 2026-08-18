"""
Import normalized Tardis derivative_ticker + liquidations CSV(.gz) into
RMV5 features.sqlite.
"""
from __future__ import annotations
import argparse, glob, sqlite3
from pathlib import Path
import pandas as pd

def ensure_db(db: Path):
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db)
    con.execute("""
        CREATE TABLE IF NOT EXISTS features (
            bucket_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            oi REAL,
            funding_rate REAL DEFAULT 0,
            long_liq_usdt REAL DEFAULT 0,
            short_liq_usdt REAL DEFAULT 0,
            taker_ratio REAL DEFAULT 1,
            top_ls_ratio REAL DEFAULT 1,
            updated_ms INTEGER NOT NULL,
            PRIMARY KEY (bucket_ms, symbol)
        )
    """)
    con.commit()
    return con

def expand(patterns):
    out = []
    for p in patterns or []:
        m = glob.glob(p)
        out.extend(m or [p])
    return sorted(set(out))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/freqtrade/user_data/v5/features.sqlite")
    ap.add_argument("--derivative-ticker", nargs="*", default=[])
    ap.add_argument("--liquidations", nargs="*", default=[])
    args = ap.parse_args()
    con = ensure_db(Path(args.db))

    for fn in expand(args.derivative_ticker):
        df = pd.read_csv(fn)
        if df.empty: continue
        df["date"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="us", utc=True)
        df["bucket"] = df["date"].dt.floor("15min")
        oi_name = "open_interest" if "open_interest" in df.columns else "openInterest"
        fr_name = "funding_rate" if "funding_rate" in df.columns else "fundingRate"
        agg = df.groupby(["symbol", "bucket"], as_index=False).agg(
            open_interest=(oi_name, "last"),
            funding_rate=(fr_name, "last"),
        )
        for _, r in agg.iterrows():
            b = int(r["bucket"].timestamp() * 1000)
            con.execute("""
                INSERT INTO features
                (bucket_ms, symbol, oi, funding_rate, long_liq_usdt, short_liq_usdt,
                 taker_ratio, top_ls_ratio, updated_ms)
                VALUES (?, ?, ?, ?, 0, 0, 1, 1, ?)
                ON CONFLICT(bucket_ms, symbol) DO UPDATE SET
                  oi=COALESCE(excluded.oi, features.oi),
                  funding_rate=excluded.funding_rate
            """, (
                b, str(r["symbol"]).upper(),
                float(r["open_interest"]) if pd.notna(r["open_interest"]) else None,
                float(r["funding_rate"]) if pd.notna(r["funding_rate"]) else 0.0,
                b,
            ))
        con.commit()
        print("derivative ticker:", fn)

    for fn in expand(args.liquidations):
        df = pd.read_csv(fn)
        if df.empty: continue
        df["date"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="us", utc=True)
        df["bucket"] = df["date"].dt.floor("15min")
        df["notional"] = pd.to_numeric(df["price"], errors="coerce") * pd.to_numeric(df["amount"], errors="coerce")
        df["long_liq"] = df["notional"].where(df["side"].astype(str).str.lower() == "sell", 0.0)
        df["short_liq"] = df["notional"].where(df["side"].astype(str).str.lower() == "buy", 0.0)
        agg = df.groupby(["symbol", "bucket"], as_index=False).agg(
            long_liq=("long_liq", "sum"),
            short_liq=("short_liq", "sum"),
        )
        for _, r in agg.iterrows():
            b = int(r["bucket"].timestamp() * 1000)
            con.execute("""
                INSERT INTO features
                (bucket_ms, symbol, oi, funding_rate, long_liq_usdt, short_liq_usdt,
                 taker_ratio, top_ls_ratio, updated_ms)
                VALUES (?, ?, NULL, 0, ?, ?, 1, 1, ?)
                ON CONFLICT(bucket_ms, symbol) DO UPDATE SET
                  long_liq_usdt=excluded.long_liq_usdt,
                  short_liq_usdt=excluded.short_liq_usdt
            """, (b, str(r["symbol"]).upper(), float(r["long_liq"]), float(r["short_liq"]), b))
        con.commit()
        print("liquidations:", fn)

    con.close()

if __name__ == "__main__":
    main()
