#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

try:
    import kagglehub
except Exception as e:
    raise SystemExit(f"kagglehub import failed: {e}")

DATASET = "bizzyvinci/coinmarketcap-historical-data"


def pick_col(cols: list[str], names: list[str]) -> str | None:
    lower = {str(c).lower(): str(c) for c in cols}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def norm_meta(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip()
    if not s:
        return ""
    obj = None
    for fn in (json.loads, ast.literal_eval):
        try:
            obj = fn(s)
            break
        except Exception:
            pass
    if isinstance(obj, dict):
        return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if isinstance(obj, (list, tuple, set)):
        return "|".join(sorted({str(x).strip().lower() for x in obj if str(x).strip()}))
    # fall back to a stable, order-insensitive split for common CMC tag strings
    parts = [x.strip().strip("'\"").lower() for x in re.split(r"[,|;]", s.strip("[]")) if x.strip()]
    return "|".join(sorted(set(parts))) if len(parts) > 1 else s.lower()


def extract_latest_version(path: str) -> int | None:
    m = re.search(r"[/\\]versions[/\\](\d+)(?:[/\\]|$)", path)
    return int(m.group(1)) if m else None


def download_coins(handle: str, output_root: Path) -> Path:
    # path= keeps this audit tiny: only metadata, never the multi-million-row historical.csv.
    p = kagglehub.dataset_download(handle, path="coins.csv", output_dir=str(output_root / handle.replace("/", "__")))
    path = Path(p)
    if path.is_dir():
        cand = path / "coins.csv"
        if cand.exists():
            return cand
        matches = list(path.rglob("coins.csv"))
        if matches:
            return matches[0]
    if path.name == "coins.csv" and path.exists():
        return path
    matches = list(output_root.rglob("coins.csv"))
    if matches:
        return max(matches, key=lambda x: x.stat().st_mtime)
    raise FileNotFoundError(f"coins.csv not found after KaggleHub returned {p}")


def summarize(version: int, path: Path) -> tuple[dict, pd.DataFrame, str | None, str | None]:
    df = pd.read_csv(path, low_memory=False)
    cols = [str(c) for c in df.columns]
    id_col = pick_col(cols, ["id", "coin_id", "cmc_id"])
    tags_col = pick_col(cols, ["tag_names", "tags", "tag_slugs"])
    category_col = pick_col(cols, ["category"])
    date_col = pick_col(cols, ["last_updated", "date_updated", "updated_at", "date_added"])

    if id_col is None:
        raise RuntimeError(f"No CMC id column in coins.csv v{version}. Columns={cols}")

    meta_col = tags_col or category_col
    work = df.copy()
    work["__id"] = pd.to_numeric(work[id_col], errors="coerce")
    work = work[work["__id"].notna()].copy()
    work["__id"] = work["__id"].astype("int64")
    if meta_col:
        work["__meta"] = work[meta_col].map(norm_meta)
    else:
        work["__meta"] = ""

    date_max = None
    if date_col:
        d = pd.to_datetime(work[date_col], errors="coerce", utc=True)
        if d.notna().any():
            date_max = str(d.max())

    sig = hashlib.sha256("\n".join(f"{i}:{m}" for i, m in sorted(zip(work["__id"], work["__meta"]))).encode("utf-8")).hexdigest()[:16]
    nonempty = work["__meta"].ne("")
    unique_tokens: set[str] = set()
    for s in work.loc[nonempty, "__meta"].astype(str):
        unique_tokens.update(x for x in s.split("|") if x)

    info = {
        "version": version,
        "rows": int(len(work)),
        "id_col": id_col,
        "meta_col": meta_col,
        "category_col": category_col,
        "tags_col": tags_col,
        "date_col": date_col,
        "max_metadata_date": date_max,
        "nonempty_meta_pct": float(nonempty.mean() * 100.0) if len(work) else 0.0,
        "distinct_meta_tokens": int(len(unique_tokens)),
        "signature": sig,
        "path": str(path),
    }
    return info, work[["__id", "__meta"]].drop_duplicates("__id", keep="last"), meta_col, date_col


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit whether old Kaggle versions preserve point-in-time CoinMarketCap category/tag metadata")
    ap.add_argument("--out", default="/freqtrade/user_data/cmc_category_audit")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dl = out / "downloads"
    dl.mkdir(parents=True, exist_ok=True)

    print("=== CMC CATEGORY HISTORY DATA-VIABILITY AUDIT ===")
    print(f"Dataset: {DATASET}")
    print("Downloads ONLY coins.csv metadata from a handful of Kaggle dataset versions.")
    print("No strategy backtest. No historical.csv download.\n")

    latest_path = download_coins(DATASET, dl)
    latest_v = extract_latest_version(str(latest_path))
    if latest_v is None:
        # output_dir can hide KaggleHub's cache version in the returned path. Ask KaggleHub once without output_dir.
        raw = kagglehub.dataset_download(DATASET, path="coins.csv")
        latest_v = extract_latest_version(str(raw))
    if latest_v is None:
        raise RuntimeError(f"Could not infer current Kaggle dataset version from {latest_path}")

    sample_versions = sorted(set(v for v in [1, max(1, latest_v // 4), max(1, latest_v // 2), max(1, (3 * latest_v) // 4), max(1, latest_v - 12), max(1, latest_v - 6), latest_v] if v <= latest_v))
    print(f"Current version: {latest_v}")
    print(f"Sample versions: {sample_versions}\n")

    infos: list[dict] = []
    frames: dict[int, pd.DataFrame] = {}
    failures: list[tuple[int, str]] = []
    for v in sample_versions:
        handle = f"{DATASET}/versions/{v}"
        try:
            path = download_coins(handle, dl / f"v{v}")
            info, frame, _, _ = summarize(v, path)
            infos.append(info)
            frames[v] = frame
            print(
                f"v{v}: rows={info['rows']:,} meta={info['meta_col']} nonempty={info['nonempty_meta_pct']:.1f}% "
                f"tokens={info['distinct_meta_tokens']:,} max_date={info['max_metadata_date']} sig={info['signature']}"
            )
        except Exception as e:
            failures.append((v, f"{type(e).__name__}: {e}"))
            print(f"v{v}: DOWNLOAD/READ FAILED: {type(e).__name__}: {e}")

    print("\nPAIRWISE METADATA CHANGE")
    pairs = []
    loaded = sorted(frames)
    for a, b in zip(loaded, loaded[1:]):
        x = frames[a].rename(columns={"__meta": "meta_a"})
        y = frames[b].rename(columns={"__meta": "meta_b"})
        z = x.merge(y, on="__id", how="inner")
        changed = z["meta_a"].ne(z["meta_b"])
        pct = float(changed.mean() * 100.0) if len(z) else float("nan")
        added = len(set(y["__id"]) - set(x["__id"]))
        removed = len(set(x["__id"]) - set(y["__id"]))
        pairs.append({"v_old": a, "v_new": b, "common_ids": len(z), "changed_meta_pct": pct, "added_ids": added, "removed_ids": removed})
        print(f"v{a} -> v{b}: common={len(z):,} changed_meta={pct:.2f}% added_ids={added:,} removed_ids={removed:,}")

    pd.DataFrame(infos).to_csv(out / "version_summary.csv", index=False)
    pd.DataFrame(pairs).to_csv(out / "pairwise_changes.csv", index=False)

    print("\nAUDIT INTERPRETATION")
    if len(infos) < 3:
        print("[INSUFFICIENT] Fewer than 3 historical metadata versions were readable. Do NOT build CCM yet.")
    else:
        meta_cols = {i.get("meta_col") for i in infos if i.get("meta_col")}
        any_change = any(p["changed_meta_pct"] > 0.1 or p["added_ids"] > 0 or p["removed_ids"] > 0 for p in pairs)
        if not meta_cols:
            print("[FAIL] coins.csv has no tags/category field usable for CMC economic-category membership. Do NOT build CCM from this dataset.")
        elif not any_change:
            print("[FAIL] Old versions appear metadata-identical. They do not demonstrate point-in-time category snapshots. Do NOT build CCM from this dataset.")
        else:
            print("[PROMISING, NOT YET PROVEN] Kaggle versions contain version-specific metadata. Next audit must map version dates and verify that tag changes are contemporaneous rather than retrospectively backfilled.")
    if failures:
        print(f"Failed sampled versions: {failures}")
    print(f"Saved: {out}/version_summary.csv, {out}/pairwise_changes.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
