#!/usr/bin/env python3
"""
0625_Prepare LoRA fine-tuning data for Qwen3.5 crash-report student model.

目标 / Goal:
  Pool human-reviewed gold (200) + Gemini silver (1827), shuffle, and split 5:2:3:
    train=1000, val=400, test=600  (total 2000; 27 held out from 2027 pool).
  合并 gold+silver，打乱后按 5:2:3 划分：train=1000, val=400, test=600。

  Silver and gold are treated as equally usable supervision (user-reviewed quality).
  silver 与 gold 视为同等可用监督（经用户抽查质量接近）。

Important / 重要:
  Fixed fields (scene/weather/lighting/collision_type) are OFFICIAL GT values
  (verified 0-mismatch vs ground_truth_full.csv + test_metadata_full.csv).
  固定字段为官方 GT 值，非 Gemini 生成。

Output / 输出:
  lora_finetune/data/
    frames_cache_pre13_post7/<video_stem>/f00.jpg ... f19.jpg
    train.jsonl / val.jsonl / test.jsonl
    train_gold.jsonl   # gold-only subset of train (optional stage-2 refine)
    split_meta.json    # per-split counts by source (gold/silver)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent
sys.path.insert(0, str(EXP_DIR))

from frame_strategies import get_offsets  # noqa: E402
from video_utils import extract_frames_at_offsets, resolve_video_path  # noqa: E402

GT_DIR = EXP_DIR / "crash_description_gt"
GOLD_FILE = GT_DIR / "crash_description_gt_review.json"
SILVER_FILE = GT_DIR / "crash_description_gt_silver.json"

DATA_DIR = HERE / "data"
DEFAULT_STRATEGY = "pre13_post7"  # 20 frames: 13 pre-onset + 7 post-onset

# Default unified split: 5:2:3 on 2000 samples (1000/400/600).
# 默认统一划分：2000 条按 5:2:3 → 1000/400/600。
DEFAULT_TRAIN = 1000
DEFAULT_VAL = 400
DEFAULT_TEST = 600

TARGET_FIELDS = [
    "scene", "weather", "lighting", "participants",
    "collision_type", "observation", "explanation", "summary",
]


def tag_source(records: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Attach source tag without mutating originals in caller's scope."""
    # 为每条记录打上来源标签
    out = []
    for r in records:
        tagged = dict(r)
        tagged["_source"] = source
        out.append(tagged)
    return out


def split_pool(
    pool: list[dict[str, Any]],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """
    Shuffle the full pool and slice train/val/test; remainder is held out.
    打乱全集后切分 train/val/test；剩余样本不使用。
    """
    rng = random.Random(seed)
    rng.shuffle(pool)
    n_need = n_train + n_val + n_test
    if n_need > len(pool):
        raise ValueError(f"split {n_need} > pool {len(pool)}")
    used = pool[:n_need]
    held = pool[n_need:]
    train = used[:n_train]
    val = used[n_train:n_train + n_val]
    test = used[n_train + n_val:n_need]
    return train, val, test, held


def extract_and_cache_frames(
    path: str, accident_time: float, offsets: list[float], frames_dir: Path
) -> list[str]:
    """Extract and cache frames; return absolute JPG paths."""
    # 抽取并缓存关键帧，返回绝对路径
    stem = Path(path).stem
    out_dir = frames_dir / stem
    expected = [out_dir / f"f{i:02d}.jpg" for i in range(len(offsets))]
    if all(p.is_file() for p in expected):
        return [str(p) for p in expected]

    vpath = resolve_video_path(path)
    if not vpath:
        return []
    frames, _times, _dur = extract_frames_at_offsets(vpath, accident_time, offsets)
    if not frames:
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for i, img in enumerate(frames):
        fp = out_dir / f"f{i:02d}.jpg"
        img.convert("RGB").save(fp, quality=90)
        saved.append(str(fp))
    return saved


def build_record(
    rec: dict[str, Any], accident_time: float, offsets: list[float], frames_dir: Path
) -> dict[str, Any] | None:
    """Build one jsonl record: frames + target + source tag."""
    # 构造一条 jsonl 记录：帧路径 + 目标 JSON + 来源
    frames = extract_and_cache_frames(rec["path"], accident_time, offsets, frames_dir)
    if not frames:
        print(f"  SKIP no frames: {rec['path']}", flush=True)
        return None
    target = {f: rec.get(f) for f in TARGET_FIELDS}
    return {
        "path": rec["path"],
        "frames": frames,
        "target": target,
        "source": rec.get("_source", "unknown"),
    }


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} -> {path}", flush=True)


def source_stats(records: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(r.get("source", "?") for r in records))


def load_accident_times() -> dict[str, float]:
    import pandas as pd

    # Official onset times stay in the data tree. / 官方 onset 时刻在数据目录。
    sys.path.insert(0, str(EXP_DIR.parent.parent))
    from crashalign_paths import accident_root

    gt = pd.read_csv(accident_root() / "ACCIDENT_dataset" / "ground_truth_full.csv")
    return {str(r["path"]): float(r["accident_time"]) for _, r in gt.iterrows()}


def process_split(
    records: list[dict[str, Any]],
    at_map: dict[str, float],
    offsets: list[float],
    frames_dir: Path,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rec in records:
        at = at_map.get(str(rec["path"]))
        if at is None:
            print(f"  SKIP no official accident_time: {rec['path']}", flush=True)
            continue
        built = build_record(rec, at, offsets, frames_dir)
        if built:
            out.append(built)
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare unified LoRA data (gold+silver pool)")
    parser.add_argument("--train", type=int, default=DEFAULT_TRAIN)
    parser.add_argument("--val", type=int, default=DEFAULT_VAL)
    parser.add_argument("--test", type=int, default=DEFAULT_TEST)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY)
    parser.add_argument("--limit", type=int, default=None, help="Debug: cap pool size before split")
    args = parser.parse_args()

    offsets = get_offsets(args.strategy)
    frames_dir = DATA_DIR / f"frames_cache_{args.strategy}"
    print(f"strategy={args.strategy} frames={len(offsets)} cache={frames_dir}", flush=True)

    gold = tag_source(json.loads(GOLD_FILE.read_text(encoding="utf-8")), "gold")
    silver = tag_source(json.loads(SILVER_FILE.read_text(encoding="utf-8")), "silver")
    pool = gold + silver
    print(f"pool: gold={len(gold)} silver={len(silver)} total={len(pool)}", flush=True)

    if args.limit is not None:
        rng = random.Random(args.seed)
        rng.shuffle(pool)
        pool = pool[:args.limit]

    train_raw, val_raw, test_raw, held = split_pool(
        pool, args.train, args.val, args.test, args.seed
    )
    print(
        f"split: train={len(train_raw)} val={len(val_raw)} test={len(test_raw)} held_out={len(held)}",
        flush=True,
    )

    at_map = load_accident_times()
    kw = dict(offsets=offsets, frames_dir=frames_dir)

    train = process_split(train_raw, at_map, **kw)
    val = process_split(val_raw, at_map, **kw)
    test = process_split(test_raw, at_map, **kw)
    train_gold = [r for r in train if r.get("source") == "gold"]
    val_gold = [r for r in val if r.get("source") == "gold"]

    write_jsonl(train, DATA_DIR / "train.jsonl")
    write_jsonl(val, DATA_DIR / "val.jsonl")
    write_jsonl(test, DATA_DIR / "test.jsonl")
    write_jsonl(train_gold, DATA_DIR / "train_gold.jsonl")
    write_jsonl(val_gold, DATA_DIR / "val_gold.jsonl")

    meta = {
        "seed": args.seed,
        "strategy": args.strategy,
        "pool": {"gold": len(gold), "silver": len(silver), "total": len(pool)},
        "split_requested": {"train": args.train, "val": args.val, "test": args.test},
        "split_written": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "train_gold": len(train_gold),
            "val_gold": len(val_gold),
            "held_out": len(held),
        },
        "source_in_split": {
            "train": source_stats(train),
            "val": source_stats(val),
            "val_gold": source_stats(val_gold),
            "test": source_stats(test),
        },
    }
    (DATA_DIR / "split_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
