from __future__ import annotations
import argparse, json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--universe", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.source).read_text())
    uni = json.loads(Path(args.universe).read_text())
    symbols = [str(x).upper() for x in uni["symbols"]]
    pairs = [f"{s[:-4]}/USDT:USDT" for s in symbols if s.endswith("USDT")]
    cfg["exchange"]["pair_whitelist"] = pairs

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2))
    print(f"Historical universe ({len(symbols)}): {','.join(symbols)}")
    print(f"Backtest config: {out}")


if __name__ == "__main__":
    main()
