#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

LEVELS_DONE_RE = re.compile(r"build confirmed 15m levels DONE levels=([\d,]+)")
SCAN_RE = re.compile(r"scan level interactions ([\d,]+) levels checked")


def safe_name(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def fmt_time(seconds: float | None) -> str:
    if seconds is None or not (seconds >= 0):
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


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
    print(f"pairs={len(pairs)} workers={workers} | no downloads", flush=True)
    print(f"Range: {args.start} .. {args.end}", flush=True)
    print("Detailed worker logs: " + str(parts / "<PAIR>" / "worker.log"), flush=True)
    print("Progress below updates in-place. ETA is approximate until enough work is observed.\n", flush=True)

    lock = threading.Lock()
    started = time.monotonic()
    stop_dashboard = threading.Event()
    state = {
        pair: {
            "started": False,
            "done": False,
            "failed": False,
            "stage": "queued",
            "levels_total": 0,
            "levels_checked": 0,
            "fraction": 0.0,
        }
        for pair in pairs
    }

    def recalc_fraction(pair: str) -> None:
        s = state[pair]
        if s["done"]:
            s["fraction"] = 1.0
            return
        stage = s["stage"]
        total = s["levels_total"]
        checked = s["levels_checked"]
        if total > 0:
            scan = min(max(checked / total, 0.0), 1.0)
            # Data/features/level construction ~15%, interaction scan ~80%, simulation/finalization ~5%.
            s["fraction"] = min(0.95, 0.15 + 0.80 * scan)
            if stage in {"simulate", "finalize"}:
                s["fraction"] = max(s["fraction"], 0.95)
        elif stage == "levels":
            s["fraction"] = 0.10
        elif stage in {"detail", "features"}:
            s["fraction"] = 0.05
        elif s["started"]:
            s["fraction"] = 0.02

    def parse_progress(pair: str, line: str) -> None:
        with lock:
            s = state[pair]
            if "load 15m START" in line:
                s["stage"] = "load15m"
            elif "prepare 15m features START" in line:
                s["stage"] = "features"
            elif "detail data START" in line:
                s["stage"] = "detail"
            elif "build confirmed 15m levels START" in line:
                s["stage"] = "levels"
            elif "scan level interactions START" in line:
                s["stage"] = "scan"
            elif "simulate detected setups START" in line:
                s["stage"] = "simulate"

            m = LEVELS_DONE_RE.search(line)
            if m:
                s["levels_total"] = int(m.group(1).replace(",", ""))
                s["stage"] = "scan"
            m = SCAN_RE.search(line)
            if m:
                s["levels_checked"] = int(m.group(1).replace(",", ""))
                s["stage"] = "scan"
            recalc_fraction(pair)

    def dashboard() -> None:
        width = 24
        while not stop_dashboard.wait(1.0):
            with lock:
                done = sum(1 for s in state.values() if s["done"])
                failed = sum(1 for s in state.values() if s["failed"])
                active = sum(1 for s in state.values() if s["started"] and not s["done"])
                work = sum(float(s["fraction"]) for s in state.values()) / len(pairs)
                known_total = sum(int(s["levels_total"]) for s in state.values())
                known_checked = sum(min(int(s["levels_checked"]), int(s["levels_total"])) for s in state.values() if s["levels_total"])

            elapsed = time.monotonic() - started
            eta = None
            if work >= 0.02:
                eta = elapsed * (1.0 - work) / work
            filled = min(width, max(0, int(round(width * work))))
            bar = "█" * filled + "░" * (width - filled)
            pct = work * 100.0
            levels_txt = f"levels {known_checked/1000:.0f}k/{known_total/1000:.0f}k" if known_total else "levels building"
            fail_txt = f" | failed {failed}" if failed else ""
            line = (
                f"\r\033[2K[{bar}] {done}/{len(pairs)} pairs | work {pct:5.1f}% | "
                f"{levels_txt} | active {active} | elapsed {fmt_time(elapsed)} | ETA ~{fmt_time(eta)}{fail_txt}"
            )
            sys.stdout.write(line)
            sys.stdout.flush()

    dash_thread = threading.Thread(target=dashboard, daemon=True)
    dash_thread.start()

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
            state[pair]["started"] = True
            state[pair]["stage"] = "starting"
            recalc_fraction(pair)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        error_lines: list[str] = []
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
                parse_progress(pair, line)
                if line.startswith("Traceback") or "RuntimeError" in line or "Error:" in line:
                    error_lines.append(line)
            rc = proc.wait()

        elapsed = time.monotonic() - t0
        with lock:
            state[pair]["done"] = True
            state[pair]["failed"] = rc != 0
            state[pair]["stage"] = "done" if rc == 0 else "failed"
            state[pair]["fraction"] = 1.0
        return index, pair, rc, elapsed, part, error_lines[-3:]

    failures = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(run_pair, i, pair) for i, pair in enumerate(pairs, 1)]
            for fut in as_completed(futures):
                index, pair, rc, elapsed, part, errs = fut.result()
                if rc != 0:
                    failures.append((pair, rc, part / "worker.log", errs))
    finally:
        stop_dashboard.set()
        dash_thread.join(timeout=2.0)
        with lock:
            done = sum(1 for s in state.values() if s["done"])
            failed_count = sum(1 for s in state.values() if s["failed"])
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()
        print(
            f"Workers finished: {done}/{len(pairs)} | failed={failed_count} | elapsed={fmt_time(time.monotonic()-started)}",
            flush=True,
        )

    if failures:
        print("\n=== WORKER FAILURES ===", flush=True)
        for pair, rc, log, errs in failures:
            print(f"{pair}: rc={rc} log={log}", flush=True)
            for e in errs:
                print(f"  {e}", flush=True)
        return 2

    print("Aggregating pair outputs...", flush=True)
    agg = [
        sys.executable, "-u", "/opt/rmv5/tools/aggregate_level_edge_parts.py",
        "--parts", str(parts),
        "--outdir", str(outdir),
    ]
    return subprocess.call(agg)


if __name__ == "__main__":
    raise SystemExit(main())
