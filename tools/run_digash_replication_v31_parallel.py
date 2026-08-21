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

PHASE_WEIGHTS = {"load": 0.08, "levels": 0.22, "scan": 0.50, "simulate": 0.20}
PHASE_ORDER = list(PHASE_WEIGHTS)


def safe(pair: str) -> str:
    return pair.replace("/", "_").replace(":", "_")


def parse_args():
    p = argparse.ArgumentParser(description="Parallel Digash replication V3.1 runner")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_replication_v31")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def fmt(s):
    if s is None or s < 0 or s > 99*3600:
        return "--:--"
    s = int(s)
    h, r = divmod(s, 3600)
    m, ss = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{ss:02d}" if h else f"{m:02d}:{ss:02d}"


def bar(p, w=24):
    p = max(0, min(1, p))
    n = int(round(p*w))
    return "[" + "█"*n + "░"*(w-n) + "]"


def pp(st):
    if st.get("done"):
        return 1.0
    ph = st.get("phase")
    if ph not in PHASE_WEIGHTS:
        return 0.0
    base = sum(PHASE_WEIGHTS[x] for x in PHASE_ORDER[:PHASE_ORDER.index(ph)])
    return min(.999, base + PHASE_WEIGHTS[ph]*max(0, min(1, st.get("frac", 0))))


def main() -> int:
    a = parse_args()
    cfg = json.loads(Path(a.config).read_text())
    pairs = list(cfg.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair_whitelist")
    workers = max(1, min(a.workers, len(pairs)))
    out = Path(a.outdir)
    parts, confs, act = out/"parts", out/"worker_configs", out/"activity"
    out.mkdir(parents=True, exist_ok=True)
    for p in (parts, confs, act):
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

    print("=== DIGASH REPLICATION V3.1 ===", flush=True)
    print(f"pairs={len(pairs)} | workers={workers} processes | CACHE ONLY", flush=True)
    print("Fidelity fixes: top-5 activity lists + top volume; one-sided breakout protorgovka; causal structure stop; correct opposing-level targets.", flush=True)
    print("No invented lifetime expiry. No historical trade-count/densities.", flush=True)

    print("\nPreparing Digash-style activity lists...", flush=True)
    alog = out/"activity.log"
    cmd = [sys.executable, "-u", "/opt/rmv5/tools/prepare_digash_activity_v31.py",
           "--config", a.config, "--datadir", a.datadir, "--outdir", str(act),
           "--start", a.start, "--end", a.end]
    with alog.open("w", encoding="utf-8") as lf:
        pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert pr.stdout is not None
        for raw in pr.stdout:
            lf.write(raw); lf.flush()
            line = raw.strip()
            if line.startswith("ACTIVITY|"):
                f = line.split("|")
                try:
                    i, n = int(f[1]), int(f[2])
                    sys.stdout.write(f"\r{bar(i/n)} activity {i}/{n}"); sys.stdout.flush()
                except Exception:
                    pass
        rc = pr.wait()
    sys.stdout.write("\r\033[K"); sys.stdout.flush()
    if rc != 0:
        print(f"Activity FAILED rc={rc}; see {alog}", flush=True)
        return rc
    print("Activity lists ready.", flush=True)

    import pandas as pd
    mf = pd.read_csv(act/"manifest.csv")
    manifest = {str(r.pair): str(r.file) for r in mf.itertuples(index=False)}
    states = {p: {"phase":"load", "frac":0.0, "done":False, "failed":False, "started":None} for p in pairs}
    lock = threading.Lock(); stop = threading.Event(); t0 = time.monotonic()

    def dashboard():
        while not stop.is_set():
            with lock:
                vals = [pp(states[p]) for p in pairs]
                work = sum(vals)/len(vals)
                done = sum(states[p]["done"] for p in pairs)
                fail = sum(states[p]["failed"] for p in pairs)
                active = sum(states[p]["started"] is not None and not states[p]["done"] for p in pairs)
                pc = {}
                for p in pairs:
                    if states[p]["started"] is not None and not states[p]["done"]:
                        pc[states[p]["phase"]] = pc.get(states[p]["phase"], 0) + 1
            el = time.monotonic()-t0
            eta = el*(1-work)/work if work >= .03 else None
            pt = ",".join(f"{k}:{v}" for k,v in sorted(pc.items()))
            sys.stdout.write("\r\033[K" + f"{bar(work)} {work*100:5.1f}% | done {done}/{len(pairs)} | active {active}/{workers} | failed {fail} | elapsed {fmt(el)} | ETA {fmt(eta)} | {pt}")
            sys.stdout.flush(); stop.wait(1)

    dt = threading.Thread(target=dashboard, daemon=True)

    def run_pair(pair):
        s = safe(pair); part = parts/s; part.mkdir(parents=True, exist_ok=True)
        cp = confs/f"{s}.json"
        cc = json.loads(json.dumps(cfg)); cc.setdefault("exchange", {})["pair_whitelist"] = [pair]
        cp.write_text(json.dumps(cc, indent=2))
        log = part/"worker.log"; af = manifest.get(pair)
        if not af:
            return pair, 98, log
        cmd = [sys.executable, "-u", "/opt/rmv5/tools/audit_digash_replication_v31.py",
               "--config", str(cp), "--datadir", a.datadir, "--activity-file", af,
               "--outdir", str(part), "--start", a.start, "--end", a.end]
        with lock:
            states[pair]["started"] = time.monotonic()
        env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
        with log.open("w", encoding="utf-8") as lf:
            pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            assert pr.stdout is not None
            for raw in pr.stdout:
                lf.write(raw); lf.flush(); line = raw.strip()
                if line.startswith("PROGRESS|"):
                    f = line.split("|")
                    if len(f) >= 4:
                        try:
                            d, tot, ph = float(f[2]), max(float(f[3]), 1), f[1]
                            with lock:
                                states[pair]["phase"] = ph; states[pair]["frac"] = min(1, d/tot)
                        except ValueError:
                            pass
            rc = pr.wait()
        with lock:
            states[pair]["done"] = True; states[pair]["failed"] = rc != 0; states[pair]["frac"] = 1
        return pair, rc, log

    print("\nRunning pair workers:", flush=True); dt.start(); fail = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fs = [ex.submit(run_pair, p) for p in pairs]
        for f in as_completed(fs):
            p, rc, l = f.result()
            if rc != 0:
                fail.append((p, rc, l))
    stop.set(); dt.join(timeout=2); sys.stdout.write("\r\033[K"); sys.stdout.flush()
    el = time.monotonic()-t0
    print(f"Workers finished: {len(pairs)-len(fail)}/{len(pairs)} ok | failed={len(fail)} | elapsed={fmt(el)}", flush=True)
    if fail:
        print("\n=== WORKER FAILURES ===", flush=True)
        for p, rc, l in fail:
            print(f"{p}: rc={rc} log={l}", flush=True)
        return 2
    print("Aggregating V3.1 replication study...", flush=True)
    return subprocess.call([sys.executable, "-u", "/opt/rmv5/tools/aggregate_digash_replication_v31.py", "--parts", str(parts), "--outdir", str(out)])


if __name__ == "__main__":
    raise SystemExit(main())
