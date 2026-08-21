#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def safe_name(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def parse_args():
    p = argparse.ArgumentParser(description="Run level-edge audit in parallel, one process per pair")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/level_edge_audit_16w")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_cfg_path = Path(args.config)
    if not base_cfg_path.exists():
        raise SystemExit(f"CONFIG_MISSING: {base_cfg_path}")

    base_cfg = json.loads(base_cfg_path.read_text())
    pairs = list(base_cfg.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise SystemExit("No pair_whitelist in config")

    workers = max(1, min(args.workers, len(pairs)))
    outdir = Path(args.outdir)
    parts = outdir / "parts"
    configs = outdir / "worker_configs"
    if parts.exists():
        shutil.rmtree(parts)
    if configs.exists():
        shutil.rmtree(configs)
    parts.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)

    print("=== PARALLEL LEVEL EDGE AUDIT ===", flush=True)
    print(f"pairs={len(pairs)} workers={workers}", flush=True)
    print("Each pair runs in a separate Python PROCESS. No market downloads.", flush=True)
    print(f"Range: {args.start} .. {args.end}", flush=True)
    print(f"Output: {outdir}\n", flush=True)

    lock = threading.Lock()
    started = time.monotonic()

    def run_pair(index: int, pair: str):
        t0 = time.monotonic()
        safe = safe_name(pair)
        part = parts / safe
        part.mkdir(parents=True, exist_ok=True)
        cfg_path = configs / f"{safe}.json"
        cfg = json.loads(json.dumps(base_cfg))
        cfg.setdefault("exchange", {})["pair_whitelist"] = [pair]
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log_path = part / "worker.log"

        cmd = [
            sys.executable, "-u", "/opt/rmv5/tools/audit_level_edge_verbose.py",
            "--config", str(cfg_path),
            "--datadir", args.datadir,
            "--outdir", str(part),
            "--start", args.start,
            "--end", args.end,
        ]

        with lock:
            print(f"[master] START {index:02d}/{len(pairs)} {pair}", flush=True)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip("\n")
                lf.write(raw)
                lf.flush()
                # Keep terminal useful: stage/progress/error lines only, not every per-pair summary line.
                if (
                    line.startswith("[stage")
                    or line.startswith("Traceback")
                    or "Error" in line
                    or "RuntimeError" in line
                    or line.startswith("[") and "detail=" in line
                ):
                    with lock:
                        print(f"[{pair}] {line}", flush=True)
            rc = proc.wait()

        elapsed = time.monotonic() - t0
        return index, pair, rc, elapsed, part

    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_pair, i, pair) for i, pair in enumerate(pairs, 1)]
        for fut in as_completed(futures):
            index, pair, rc, elapsed, part = fut.result()
            done += 1
            status = "DONE" if rc == 0 else f"FAILED rc={rc}"
            with lock:
                print(
                    f"[master] {status} {index:02d}/{len(pairs)} {pair} "
                    f"[{elapsed:.1f}s] | completed={done}/{len(pairs)}",
                    flush=True,
                )
            if rc != 0:
                failures.append((pair, rc, part / "worker.log"))

    if failures:
        print("\n=== WORKER FAILURES ===", flush=True)
        for pair, rc, log in failures:
            print(f"{pair}: rc={rc} log={log}", flush=True)
        return 2

    print(f"\n[master] all {len(pairs)} pairs completed in {time.monotonic()-started:.1f}s", flush=True)
    print("[master] aggregating pair outputs...", flush=True)

    agg = [
        sys.executable, "-u", "/opt/rmv5/tools/aggregate_level_edge_parts.py",
        "--parts", str(parts),
        "--outdir", str(outdir),
    ]
    return subprocess.call(agg)


if __name__ == "__main__":
    raise SystemExit(main())
