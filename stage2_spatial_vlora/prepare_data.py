#!/usr/bin/env python3
"""
0917_Prepare GT-onset spatial data with integerized coordinates.

Uses annotated accident_time (not predicted T) to extract the onset frame,
reuses the 0914 120/40/40 split for fair comparison, and writes answers as
integer grid coords in [0, 1000].
使用标注事故时刻抽 onset 帧（非预测 T），复用 0914 划分，监督为 0–1000 整数坐标。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))  # local video_utils.py / 本目录的抽帧工具
from crashalign_paths import accident_root  # noqa: E402
from video_utils import extract_frames_at_offsets, resolve_video_path  # noqa: E402

# GT csv stays in the data tree. The 120/40/40 split file is copied into this package.
# 标注在数据目录。120/40/40 划分文件已放进本代码包。
ROOT = accident_root()
GT_CSV = ROOT / "test_200_GT" / "ground_truth_200.csv"
_PACKAGED_SPLIT = HERE.parent / "stage2_spatial_lora" / "data" / "split_meta.json"
_ORIGINAL_SPLIT = ROOT / "0914_spatial_lora" / "data" / "split_meta.json"
SPLIT_META = _PACKAGED_SPLIT if _PACKAGED_SPLIT.is_file() else _ORIGINAL_SPLIT
DATA_DIR = HERE / "data"
FRAME_DIR = HERE / "frames_gt_onset"
COORD_SCALE = 1000  # integer grid / 整数网格


def to_int_coord(v: float, scale: int = COORD_SCALE) -> int:
    """Map normalized [0,1] float to integer grid. / 归一化坐标 → 整数网格。"""
    return int(max(0, min(scale, round(float(v) * scale))))


def load_gt() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with GT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["path"]] = {
                "center_x": float(row["center_x"]),
                "center_y": float(row["center_y"]),
                "accident_time": float(row["accident_time"]),
                "type": row.get("type", ""),
            }
    return out


def extract_gt_onset(path: str, t_gt: float, stem: str) -> Path | None:
    """Extract and cache the single GT-onset frame. / 抽取并缓存标注 onset 单帧。"""
    out = FRAME_DIR / f"{stem}.jpg"
    if out.is_file():
        return out
    vpath = resolve_video_path(path)
    if vpath is None:
        return None
    frames, _times, _dur = extract_frames_at_offsets(vpath, t_gt, offsets=[0.0])
    if not frames:
        return None
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, quality=95)
    return out


def make_record(path: str, gt: dict[str, Any], frame: Path) -> dict[str, Any]:
    x_i = to_int_coord(gt["center_x"])
    y_i = to_int_coord(gt["center_y"])
    return {
        "path": path,
        "frames": [str(frame)],
        "accident_time_gt": gt["accident_time"],
        "target": {
            "x_int": x_i,
            "y_int": y_i,
            "center_x": gt["center_x"],
            "center_y": gt["center_y"],
            "answer": f"{x_i} {y_i}",
        },
        "meta": {"type_gt": gt.get("type", "")},
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_meta", type=str, default=str(SPLIT_META))
    args = parser.parse_args()

    gt = load_gt()
    meta = json.loads(Path(args.split_meta).read_text(encoding="utf-8"))
    splits: dict[str, list[str]] = meta["splits"]

    counts = {}
    missing = []
    for name, paths in splits.items():
        recs = []
        for p in paths:
            if p not in gt:
                missing.append(p)
                continue
            stem = Path(p).stem
            frame = extract_gt_onset(p, gt[p]["accident_time"], stem)
            if frame is None:
                missing.append(p)
                continue
            recs.append(make_record(p, gt[p], frame))
        out = DATA_DIR / f"{name}.jsonl"
        write_jsonl(out, recs)
        counts[name] = len(recs)
        print(f"wrote {out} n={len(recs)}", flush=True)

    summary = {
        "coord_scale": COORD_SCALE,
        "onset": "GT accident_time",
        "split_source": str(args.split_meta),
        "counts": counts,
        "missing": missing,
        "prompt": "Locate the collision point in the image.",
    }
    (DATA_DIR / "split_meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"counts={counts} missing={len(missing)}", flush=True)


if __name__ == "__main__":
    main()
