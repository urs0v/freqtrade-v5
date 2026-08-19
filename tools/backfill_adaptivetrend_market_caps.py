#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DEFAULT_DB = "/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
PUBLIC_BASE = "https://api.coingecko.com/api/v3"
DEMO_BASE = "https://api.coingecko.com/api/v3"
PRO_BASE = "https://pro-api.coingecko.com/api/v3"
MIN_USABLE_CAP_SYMBOLS = 75

OVERRIDES = {
    "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin", "XRP": "ripple", "SOL": "solana",
    "DOGE": "dogecoin", "ADA": "cardano", "TRX": "tron", "LINK": "chainlink", "LTC": "litecoin",
    "BCH": "bitcoin-cash", "ETC": "ethereum-classic", "DOT": "polkadot", "AVAX": "avalanche-2",
    "ATOM": "cosmos", "NEAR": "near", "UNI": "uniswap", "AAVE": "aave", "XLM": "stellar",
    "FIL": "filecoin", "MATIC": "matic-network", "POL": "matic-network", "APT": "aptos", "ARB": "arbitrum",
    "OP": "optimism", "SUI": "sui", "INJ": "injective-protocol", "ICP": "internet-computer",
    "HBAR": "hedera-hashgraph", "ALGO": "algorand", "VET": "vechain", "EOS": "eos", "XTZ": "tezos",
    "MKR": "maker", "CRV": "curve-dao-token", "SNX": "havven", "COMP": "compound-governance-token",
    "SUSHI": "sushi", "YFI": "yearn-finance", "ZEC": "zcash", "DASH": "dash", "KSM": "kusama",
    "RUNE": "thorchain", "GRT": "the-graph", "EGLD": "elrond-erd-2", "FLOW": "flow", "AXS": "axie-infinity",
    "SAND": "the-sandbox", "MANA": "decentraland", "APE": "apecoin", "LDO": "lido-dao", "STX": "blockstack",
    "IMX": "immutable-x", "RNDR": "render-token", "RENDER": "render-token", "FET": "fetch-ai", "TAO": "bittensor",
    "THETA": "theta-token", "KAVA": "kava", "IOTA": "iota", "NEO": "neo", "QTUM": "qtum",
    "ONT": "ontology", "ZIL": "zilliqa", "BAT": "basic-attention-token", "ENJ": "enjincoin",
    "CHZ": "chiliz", "1INCH": "1inch", "DYDX": "dydx-chain", "GMX": "gmx", "WLD": "worldcoin-wld",
    "PEPE": "pepe", "SHIB": "shiba-inu", "FLOKI": "floki", "BONK": "bonk", "WIF": "dogwifcoin",
}


def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS market_caps (
            symbol TEXT NOT NULL,
            cg_id TEXT NOT NULL,
            ts_ms INTEGER NOT NULL,
            market_cap REAL NOT NULL,
            PRIMARY KEY(symbol, ts_ms)
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_market_caps_time ON market_caps(ts_ms)")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS coingecko_mapping (
            symbol TEXT PRIMARY KEY,
            base_symbol TEXT NOT NULL,
            cg_id TEXT,
            method TEXT NOT NULL,
            status TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    con.commit()


def norm_base(symbol: str) -> str:
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    for p in ("1000000", "10000", "1000"):
        if base.startswith(p) and len(base) > len(p):
            base = base[len(p):]
            break
    return base.upper()


def get_json(session: requests.Session, url: str, params: dict, headers: dict, retries: int = 8) -> dict | list:
    delay = 3.0
    for _ in range(retries):
        r = session.get(url, params=params, headers=headers, timeout=60)
        if r.status_code == 429:
            retry = float(r.headers.get("Retry-After", delay))
            time.sleep(max(retry, delay))
            delay = min(delay * 1.8, 90.0)
            continue
        if r.status_code in {500, 502, 503, 504}:
            time.sleep(delay)
            delay = min(delay * 1.8, 60.0)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"CoinGecko request failed after retries: {url}")


def build_mapping(session: requests.Session, api_base: str, headers: dict, symbols: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    ranked: dict[str, tuple[str, float]] = {}
    for page in range(1, 13):
        data = get_json(
            session,
            f"{api_base}/coins/markets",
            {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page, "sparkline": "false"},
            headers,
        )
        if not isinstance(data, list) or not data:
            break
        for row in data:
            sym = str(row.get("symbol", "")).upper()
            cid = str(row.get("id", ""))
            mc = float(row.get("market_cap") or 0.0)
            if sym and cid and (sym not in ranked or mc > ranked[sym][1]):
                ranked[sym] = (cid, mc)
        time.sleep(1.0)

    all_coins = get_json(session, f"{api_base}/coins/list", {"include_platform": "false"}, headers)
    by_symbol: dict[str, list[str]] = {}
    if isinstance(all_coins, list):
        for row in all_coins:
            sym = str(row.get("symbol", "")).upper()
            cid = str(row.get("id", ""))
            if sym and cid:
                by_symbol.setdefault(sym, []).append(cid)

    mapping: dict[str, str] = {}
    methods: dict[str, str] = {}
    for symbol in symbols:
        base = norm_base(symbol)
        if base in OVERRIDES:
            mapping[symbol] = OVERRIDES[base]
            methods[symbol] = "override"
        elif base in ranked:
            mapping[symbol] = ranked[base][0]
            methods[symbol] = "highest_current_market_cap_for_symbol"
        elif len(by_symbol.get(base, [])) == 1:
            mapping[symbol] = by_symbol[base][0]
            methods[symbol] = "unique_symbol"
    return mapping, methods


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill historical CoinGecko market caps for AdaptiveTrend point-in-time universe")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--start", default="2020-12-01")
    ap.add_argument("--end", default="2025-01-02")
    ap.add_argument("--min-interval", type=float, default=6.2, help="Seconds between historical calls in keyless mode")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise RuntimeError(f"Missing core DB; run Binance backfill first: {db}")
    con = sqlite3.connect(db, timeout=60)
    ensure_schema(con)
    symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM candles ORDER BY symbol")]
    if not symbols:
        raise RuntimeError("No Binance candle symbols in DB")

    api_key = os.environ.get("COINGECKO_API_KEY", "").strip()
    plan = os.environ.get("COINGECKO_PLAN", "public").strip().lower()
    if api_key and plan == "pro":
        api_base = PRO_BASE
        headers = {"x-cg-pro-api-key": api_key}
        interval = min(args.min_interval, 1.2)
    elif api_key:
        api_base = DEMO_BASE
        headers = {"x-cg-demo-api-key": api_key}
        interval = min(args.min_interval, 2.2)
    else:
        api_base = PUBLIC_BASE
        headers = {}
        interval = args.min_interval
    headers["User-Agent"] = "freqtrade-v5-adaptivetrend-replication/1.0"

    session = requests.Session()
    print(f"Building CoinGecko ID mapping for {len(symbols)} historical Binance symbols...", flush=True)
    mapping, methods = build_mapping(session, api_base, headers, symbols)
    unresolved = [s for s in symbols if s not in mapping]
    now = datetime.now(timezone.utc).isoformat()
    for s in symbols:
        con.execute(
            "INSERT OR REPLACE INTO coingecko_mapping(symbol,base_symbol,cg_id,method,status,updated_at) VALUES (?,?,?,?,?,?)",
            (s, norm_base(s), mapping.get(s), methods.get(s, "unresolved"), "mapped" if s in mapping else "unresolved", now),
        )
    con.commit()
    print(f"Mapped={len(mapping)} / {len(symbols)} | unresolved={len(unresolved)}", flush=True)
    if unresolved:
        print("Unresolved sample:", ", ".join(unresolved[:30]), flush=True)

    already = {r[0] for r in con.execute("SELECT DISTINCT symbol FROM market_caps")}
    todo = [s for s in symbols if s in mapping and (args.force or s not in already)]
    print(f"Historical market-cap series to fetch: {len(todo)} | interval~{interval:.1f}s", flush=True)

    successes = 0
    failures = 0
    started = time.monotonic()
    for i, symbol in enumerate(todo, 1):
        cid = mapping[symbol]
        try:
            # Do not force interval=daily in public/keyless mode. A >90-day range is automatically daily,
            # and omitting the interval avoids plan-specific restrictions.
            data = get_json(
                session,
                f"{api_base}/coins/{cid}/market_chart/range",
                {"vs_currency": "usd", "from": args.start, "to": args.end},
                headers,
            )
            caps = data.get("market_caps", []) if isinstance(data, dict) else []
            rows = []
            for item in caps:
                if not isinstance(item, list) or len(item) < 2:
                    continue
                ts, cap = item[0], item[1]
                if cap is None:
                    continue
                rows.append((symbol, cid, int(ts), float(cap)))
            if rows:
                con.executemany("INSERT OR REPLACE INTO market_caps(symbol,cg_id,ts_ms,market_cap) VALUES (?,?,?,?)", rows)
                con.commit()
                successes += 1
                status = f"{len(rows):,} rows"
            else:
                failures += 1
                status = "NO DATA"
        except Exception as exc:
            failures += 1
            status = f"ERROR {type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        print(f"[{i:03d}/{len(todo):03d}] {symbol:<16} -> {cid:<28} {status} | {elapsed/60:.1f}m", flush=True)
        if i < len(todo):
            time.sleep(interval)

    cap_symbols = con.execute("SELECT COUNT(DISTINCT symbol) FROM market_caps").fetchone()[0]
    cap_rows = con.execute("SELECT COUNT(*) FROM market_caps").fetchone()[0]
    print("\n=== MARKET CAP DATA DONE ===")
    print(f"mapped_ids={len(mapping)} | cap_symbols={cap_symbols} | cap_rows={cap_rows:,} | fetch_failures={failures}")
    print(f"DB: {db}")
    # Individual old/delisted IDs can legitimately have no CoinGecko history. The backtester
    # enforces a stricter per-month universe-size requirement, so partial mapping is acceptable here.
    return 0 if cap_symbols >= MIN_USABLE_CAP_SYMBOLS else 2


if __name__ == "__main__":
    raise SystemExit(main())
