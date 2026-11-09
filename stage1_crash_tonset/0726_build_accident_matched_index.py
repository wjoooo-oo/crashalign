#!/usr/bin/env python3
"""Finalize matched / excluded lists from cached window centers.

After embeddings exist, re-check positive/negative placement on real `times`
and rewrite {split}_matched.jsonl + excluded_final.jsonl.

中文：在真实窗口中心上复核定长正/负样本是否可放置，并写出最终 matched 清单。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labels import CLEAN_END_MARGIN  # noqa: E402


def _load_sampling():
    """Load 0726_sampling_matched.py (name starts with digits)."""
    path = HERE / "0726_sampling_matched.py"
    spec = importlib.util.spec_from_file_location("sampling_matched", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_samp = _load_sampling()
positive_starts = _samp.positive_starts
negative_starts = _samp.negative_starts


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]):
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(HERE / "data_accident_matched"))
    p.add_argument("--cache_dir", default=str(HERE / "cache_emb_accident"))
    p.add_argument("--fixed_len", type=int, default=8)
    p.add_argument("--window_sec", type=float, default=2.0)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    L = args.fixed_len

    matched = {"train": [], "val": [], "test": []}
    excluded = []
    stats = defaultdict(int)

    for split in ("train", "val", "test"):
        rows = load_jsonl(data_dir / f"{split}_all.jsonl")
        for row in rows:
            vid = row["video_id"]
            npz = cache_dir / split / f"{vid}.npz"
            onset = float(row["accident_time"])
            if not npz.exists():
                # Only prelim-matched videos are extracted; others stay excluded.
                # 仅预筛选 matched 会抽 embedding；其余保持 excluded。
                if row.get("is_matched"):
                    row = dict(row)
                    row["is_matched"] = False
                    row["exclude_reasons"] = list(row.get("exclude_reasons") or []) + [
                        "missing_embedding"
                    ]
                    excluded.append(row)
                    stats["missing_embedding"] += 1
                else:
                    excluded.append(row)
                    stats["prelim_excluded"] += 1
                continue

            with np.load(npz, allow_pickle=True) as z:
                times = z["times"].astype(np.float32)
                t_gt = float(z["t_gt"])
            if abs(t_gt - onset) > 1e-2 and t_gt > 0:
                onset = t_gt

            pos = positive_starts(times, onset, L, window_sec=args.window_sec)
            neg = negative_starts(times, onset, L, end_margin=CLEAN_END_MARGIN)
            row = dict(row)
            row["n_pos_starts"] = len(pos)
            row["n_neg_starts"] = len(neg)
            row["n_windows"] = int(len(times))
            if pos and neg:
                row["is_matched"] = True
                row["exclude_reasons"] = []
                matched[split].append(row)
                stats["matched"] += 1
            else:
                reasons = []
                if not pos:
                    reasons.append("no_positive_placement")
                if not neg:
                    reasons.append("no_clean_normal_negative")
                row["is_matched"] = False
                row["exclude_reasons"] = reasons
                excluded.append(row)
                for r in reasons:
                    stats[r] += 1

    for split in ("train", "val", "test"):
        write_jsonl(data_dir / f"{split}_matched.jsonl", matched[split])
        crash_rows = []
        for r in matched[split]:
            crash_rows.append(
                {
                    "video_id": r["video_id"],
                    "path": r["path"],
                    "is_crash": 1,
                    "source": "accident",
                    "accident_time": r["accident_time"],
                    "source_video_id": r["source_video_id"],
                    "duration": r.get("duration"),
                    "n_pos_starts": r["n_pos_starts"],
                    "n_neg_starts": r["n_neg_starts"],
                }
            )
        write_jsonl(data_dir / f"{split}_crash.jsonl", crash_rows)

    write_jsonl(data_dir / "excluded_final.jsonl", excluded)

    meta_path = data_dir / "split_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["final_matched"] = {s: len(matched[s]) for s in matched}
    meta["final_excluded"] = len(excluded)
    meta["final_stats"] = dict(stats)
    meta["fixed_len"] = L
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({"final_matched": meta["final_matched"], "final_excluded": len(excluded), "stats": dict(stats)}, indent=2))


if __name__ == "__main__":
    main()
