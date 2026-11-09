#!/usr/bin/env python3
"""
0914_Prepare spatial-grounding QLoRA data from original qwen35_spatial onset frames.

Uses the SAME predicted-onset single-frame input as qwen35_spatial/, with GT
(center_x, center_y) as supervision. Split is confined to the 200-GT subset.
使用与 qwen35_spatial/ 相同的预测 onset 单帧输入，以 GT 坐标为监督；划分仅在 200-GT 上。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from crashalign_paths import accident_root  # noqa: E402

# Onset frames and GT csv stay in the data tree. / 起始帧和标注仍在数据目录。
ROOT = accident_root()
ONSET_DIR = ROOT / "test_200_GT" / "predicted_onset_frames"
ONSET_INDEX = ONSET_DIR / "index.csv"
GT_CSV = ROOT / "test_200_GT" / "ground_truth_200.csv"
META_CSV = ROOT / "labeled_little" / "labeled_little" / "ground_truth.csv"
DATA_DIR = HERE / "data"

DEFAULT_TRAIN = 120
DEFAULT_VAL = 40
DEFAULT_TEST = 40
DEFAULT_SEED = 42


def load_gt() -> dict[str, dict[str, Any]]:
    """Load spatial GT keyed by video path. / 按视频路径加载空间 GT。"""
    out: dict[str, dict[str, Any]] = {}
    with GT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["path"]] = {
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "type": row.get("type", ""),
                "accident_time": float(row["accident_time"]),
            }
    # Optional scene_layout from labeled_little (same 200 set).
    # 可选：从 labeled_little 读取 scene_layout（同一 200 集合）。
    if META_CSV.is_file():
        with META_CSV.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["path"] in out and "scene_layout" in row:
                    out[row["path"]]["scene_layout"] = row["scene_layout"]
    return out


def load_onset_index() -> dict[str, dict[str, Any]]:
    """Map path -> onset image metadata. / 路径到 onset 图像元数据。"""
    out: dict[str, dict[str, Any]] = {}
    with ONSET_INDEX.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img = ONSET_DIR / row["image"]
            out[row["path"]] = {
                "image": str(img),
                "pred_accident_time": float(row["pred_accident_time"]),
                "frame_idx": int(float(row["frame_idx"])),
            }
    return out


def split_paths(
    paths: list[str],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> dict[str, list[str]]:
    """Shuffle and split paths 60/20/20. / 打乱后按 60/20/20 划分。"""
    rng = random.Random(seed)
    items = list(paths)
    rng.shuffle(items)
    need = n_train + n_val + n_test
    if len(items) < need:
        raise ValueError(f"Need {need} paths, have {len(items)}")
    return {
        "train": items[:n_train],
        "val": items[n_train : n_train + n_val],
        "test": items[n_train + n_val : n_train + n_val + n_test],
    }


def make_record(path: str, gt: dict[str, Any], onset: dict[str, Any]) -> dict[str, Any]:
    """One jsonl sample: single onset frame + GT point target."""
    # 单条样本：onset 单帧 + GT 坐标目标
    return {
        "path": path,
        "frames": [onset["image"]],
        "pred_accident_time": onset["pred_accident_time"],
        "scene_layout": gt.get("scene_layout", ""),
        "target": {
            "center_x": round(float(gt["center_x"]), 4),
            "center_y": round(float(gt["center_y"]), 4),
            "confidence": 0.9,
        },
        "meta": {
            "type_gt": gt.get("type", ""),
            "accident_time_gt": gt.get("accident_time"),
            "frame_idx": onset["frame_idx"],
        },
    }


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare spatial QLoRA jsonl from onset frames")
    parser.add_argument("--n_train", type=int, default=DEFAULT_TRAIN)
    parser.add_argument("--n_val", type=int, default=DEFAULT_VAL)
    parser.add_argument("--n_test", type=int, default=DEFAULT_TEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    gt = load_gt()
    onset = load_onset_index()
    common = sorted(set(gt) & set(onset))
    missing_img = [p for p in common if not Path(onset[p]["image"]).is_file()]
    if missing_img:
        raise FileNotFoundError(f"{len(missing_img)} onset images missing, e.g. {missing_img[0]}")

    splits = split_paths(common, args.n_train, args.n_val, args.n_test, args.seed)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    counts = {}
    for name, paths in splits.items():
        recs = [make_record(p, gt[p], onset[p]) for p in paths]
        out = DATA_DIR / f"{name}.jsonl"
        write_jsonl(out, recs)
        counts[name] = len(recs)
        print(f"wrote {out}  n={len(recs)}", flush=True)

    meta = {
        "seed": args.seed,
        "counts": counts,
        "onset_dir": str(ONSET_DIR),
        "gt_csv": str(GT_CSV),
        "note": "Input frames = predicted onset (qwen35_spatial pipeline); targets = GT coords.",
        "splits": splits,
    }
    meta_path = DATA_DIR / "split_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {meta_path}", flush=True)
    print(f"pool={len(common)}  train/val/test={counts}", flush=True)


if __name__ == "__main__":
    main()
