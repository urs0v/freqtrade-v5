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

PHASE_WEIGHTS = {
    "load": 0.05,
    "zones": 0.10,
    "levels": 0.35,
    "consolidations": 0.15,
    "structure": 0.20,
    "simulate": 0.15,
}
PHASE_ORDER = list(PHASE_WEIGHTS)

def safe_name(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")

def parse_args():
    p = argparse.ArgumentParser(description="16-process Level/Structure V2 orchestrator")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/level_structure_v2")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()

def fmt_time(seconds: float | None) -> str:
    if seconds is None or not (seconds >= 0) or seconds > 99*3600:
        return "--:--"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"

def bar(pct: float, width: int = 24) -> str:
    pct = max(0.0, min(1.0, pct))
    n = int(round(pct * width))
    return "[" + "█"*n + "░"*(width-n) + "]"

def pair_progress(st: dict) -> float:
    if st.get("done"):
        return 1.0
    phase = st.get("phase")
    if phase not in PHASE_WEIGHTS:
        return 0.0
    base = sum(PHASE_WEIGHTS[p] for p in PHASE_ORDER[:PHASE_ORDER.index(phase)])
    frac = st.get("frac", 0.0)
    return min(0.999, base + PHASE_WEIGHTS[phase] * max(0.0, min(1.0, frac)))

def main() -> int:
    a = parse_args()
    base_cfg = json.loads(Path(a.config).read_text())
    pairs = list(base_cfg.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair_whitelist")
    workers = max(1, min(a.workers, len(pairs)))

    outdir = Path(a.outdir)
    parts = outdir / "parts"
    configs = outdir / "worker_configs"
    activity_dir = outdir / "activity"
    for p in (parts, configs, activity_dir):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== LEVEL / STRUCTURE EDGE V2 ===", flush=True)
    print(f"pairs={len(pairs)} | workers={workers} processes | cache-only market data", flush=True)
    print("Families: level break, break-retest, consolidation break, confirmed bounce, sweep/reclaim, structure break-retest.", flush=True)
    print("Causal rules; no parameter optimization in this run. Detailed logs stay under parts/<PAIR>/worker.log.", flush=True)
    print("Activity ranking is cross-sectional inside the configured 20-pair universe; trade-count is not available in OHLCV and is not faked.", flush=True)

    # Build a single cross-sectional activity universe before pair workers start.
    print("\nPreparing causal cross-sectional activity ranks...", flush=True)
    activity_log = outdir / "activity.log"
    cmd = [
        sys.executable, "-u", "/opt/rmv5/tools/prepare_level_structure_v2_activity.py",
        "--config", a.config, "--datadir", a.datadir, "--outdir", str(activity_dir),
        "--start", a.start, "--end", a.end,
    ]
    with activity_log.open("w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for raw in proc.stdout:
            lf.write(raw); lf.flush()
            line = raw.strip()
            if line.startswith("ACTIVITY|"):
                fields = line.split("|")
                try:
                    i, n = int(fields[1]), int(fields[2])
                    sys.stdout.write(f"\r{bar(i/n)} activity {i}/{n} pairs")
                    sys.stdout.flush()
                except Exception:
                    pass
        rc = proc.wait()
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
    if rc != 0:
        print(f"Activity preparation FAILED rc={rc}; see {activity_log}", flush=True)
        return rc
    print(f"Activity ranks ready for {len(pairs)} pairs.", flush=True)

    manifest = {}
    import pandas as pd
    mf = pd.read_csv(activity_dir / "manifest.csv")
    for r in mf.itertuples(index=False):
        manifest[str(r.pair)] = str(r.file)

    states = {
        pair: {"phase": "load", "frac": 0.0, "done": False, "failed": False, "started": None}
        for pair in pairs
    }
    lock = threading.Lock()
    start_workers = time.monotonic()
    stop_dashboard = threading.Event()

    def dashboard():
        while not stop_dashboard.is_set():
            with lock:
                vals = [pair_progress(states[p]) for p in pairs]
                work = sum(vals) / len(vals)
                done = sum(1 for p in pairs if states[p]["done"])
                failed = sum(1 for p in pairs if states[p]["failed"])
                active = sum(1 for p in pairs if states[p]["started"] is not None and not states[p]["done"])
                phase_counts = {}
                for p in pairs:
                    if states[p]["started"] is not None and not states[p]["done"]:
                        ph = states[p].get("phase", "?")
                        phase_counts[ph] = phase_counts.get(ph, 0) + 1
            elapsed = time.monotonic() - start_workers
            eta = elapsed * (1.0-work) / work if work >= 0.03 else None
            phase_txt = ",".join(f"{k}:{v}" for k,v in sorted(phase_counts.items()))
            text = (
                f"{bar(work)} {work*100:5.1f}% | done {done}/{len(pairs)} | active {active}/{workers} "
                f"| failed {failed} | elapsed {fmt_time(elapsed)} | ETA {fmt_time(eta)} | {phase_txt}"
            )
            sys.stdout.write("\r\033[K" + text)
            sys.stdout.flush()
            stop_dashboard.wait(1.0)

    dash_thread = threading.Thread(target=dashboard, daemon=True)

    def run_pair(index: int, pair: str):
        safe = safe_name(pair)
        part = parts / safe
        part.mkdir(parents=True, exist_ok=True)
        cfg_path = configs / f"{safe}.json"
        cfg = json.loads(json.dumps(base_cfg))
        cfg.setdefault("exchange", {})["pair_whitelist"] = [pair]
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log_path = part / "worker.log"
        afile = manifest.get(pair)
        if not afile:
            return pair, 98, log_path

        cmd = [
            sys.executable, "-u", "/opt/rmv5/tools/audit_level_structure_v2.py",
            "--config", str(cfg_path), "--datadir", a.datadir,
            "--activity-file", afile, "--outdir", str(part),
            "--start", a.start, "--end", a.end,
        ]
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        with lock:
            states[pair]["started"] = time.monotonic()

        with log_path.open("w", encoding="utf-8") as lf:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env
            )
            assert proc.stdout is not None
            for raw in proc.stdout:
                lf.write(raw); lf.flush()
                line = raw.strip()
                if line.startswith("PROGRESS|"):
                    f = line.split("|")
                    if len(f) >= 4:
                        phase = f[1]
                        try:
                            done = float(f[2]); total = max(float(f[3]), 1.0)
                            with lock:
                                states[pair]["phase"] = phase
                                states[pair]["frac"] = min(1.0, done/total)
                        except ValueError:
                            pass
            rc = proc.wait()
        with lock:
            states[pair]["done"] = True
            states[pair]["failed"] = rc != 0
            states[pair]["frac"] = 1.0
        return pair, rc, log_path

    print("\nRunning pair workers:", flush=True)
    dash_thread.start()
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(run_pair, i, pair) for i, pair in enumerate(pairs, 1)]
        for fut in as_completed(futures):
            pair, rc, log_path = fut.result()
            if rc != 0:
                failures.append((pair, rc, log_path))

    stop_dashboard.set()
    dash_thread.join(timeout=2)
    elapsed = time.monotonic() - start_workers
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
    print(f"Workers finished: {len(pairs)-len(failures)}/{len(pairs)} ok | failed={len(failures)} | elapsed={fmt_time(elapsed)}", flush=True)

    if failures:
        print("\n=== WORKER FAILURES ===", flush=True)
        for pair, rc, log_path in failures:
            print(f"{pair}: rc={rc} log={log_path}", flush=True)
        return 2

    print("Aggregating event study...", flush=True)
    cmd = [
        sys.executable, "-u", "/opt/rmv5/tools/aggregate_level_structure_v2.py",
        "--parts", str(parts), "--outdir", str(outdir),
    ]
    return subprocess.call(cmd)

if __name__ == "__main__":
    raise SystemExit(main())
