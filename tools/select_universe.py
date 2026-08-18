from __future__ import annotations
import argparse, json, urllib.request
from pathlib import Path

REST = "https://fapi.binance.com"
STABLES = {"USDC", "USDE", "FDUSD", "DAI", "TUSD", "USDP", "BUSD"}

def get_json(url: str):
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode("utf-8"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output-config", required=True)
    ap.add_argument("--universe", required=True)
    args = ap.parse_args()

    info = get_json(REST + "/fapi/v1/exchangeInfo")
    tickers = get_json(REST + "/fapi/v1/ticker/24hr")
    qvol = {x["symbol"]: float(x.get("quoteVolume", 0.0)) for x in tickers}

    eligible = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING": continue
        if s.get("contractType") != "PERPETUAL": continue
        if s.get("quoteAsset") != "USDT": continue
        if s.get("underlyingType") != "COIN": continue
        if s.get("baseAsset") in STABLES: continue
        eligible.append(s["symbol"])

    symbols = sorted(eligible, key=lambda x: qvol.get(x, 0), reverse=True)[:args.top]
    pairs = [f"{s[:-4]}/USDT:USDT" for s in symbols if s.endswith("USDT")]

    base = json.loads(Path(args.base_config).read_text())
    base["exchange"]["pair_whitelist"] = pairs

    Path(args.output_config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_config).write_text(json.dumps(base, indent=2))
    Path(args.universe).parent.mkdir(parents=True, exist_ok=True)
    Path(args.universe).write_text(json.dumps({"symbols": symbols, "pairs": pairs}, indent=2))
    print(f"Selected {len(symbols)} crypto perpetuals:")
    print(", ".join(symbols))

if __name__ == "__main__":
    main()
