#!/usr/bin/env python3
"""Build ACCIDENT-only source-video split and matched pos/neg recipe lists.

Protocol / 协议:
  1. Split by YouTube source id *before* clip extraction (no cross-split leakage).
     按 YouTube 源视频 ID 划分，再抽取 clip，避免泄漏。
  2. Fixed length L windows, no padding.
  3. Positive: contains annotated onset; train=random onset placement, val/test=hash.
  4. Clean-normal negative: same source mp4; last window center <= onset - 3.0 s;
     does not overlap pre-crash safety region; all windows labeled normal.
  5. Videos without a valid same-video clean-normal are excluded and listed separately.

Preliminary eligibility uses probed duration; definitive matching is done later on
cached window centers (0726_build_accident_matched_index.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from crashalign_paths import accident_root  # noqa: E402

# ACCIDENT csv and videos stay in the data tree. / 标注和视频留在数据目录。
ROOT = accident_root()
GT_CSV = ROOT / "ACCIDENT_dataset" / "ground_truth_full.csv"
VIDEO_DIR = ROOT / "test_videos" / "videos"
PROBE = HERE / "data_accident_matched" / "_probe_durations.json"

from labels import CLEAN_END_MARGIN  # noqa: E402


def youtube_source_id(video_id: str) -> str:
    """Strip trailing _NN clip index → YouTube-level source id.

    去掉末尾 _NN 片段编号，得到 YouTube 级源 ID。
    """
    m = re.match(r"^(.+)_(\d{2})$", video_id)
    return m.group(1) if m else video_id


def can_place_positive(onset: float, duration: float, L: int, window_sec: float, stride: float) -> bool:
    """Whether some L-window clip can cover the onset time.

    是否存在长度为 L 的窗口序列覆盖 onset。
    """
    half = window_sec / 2.0
    span = (L - 1) * stride
    lo = max(half, onset - span - half)
    hi = min(duration - half - span, onset + half)
    return lo <= hi + 1e-9


def can_place_negative(onset: float, L: int, window_sec: float, stride: float) -> bool:
    """Whether an L-window clean-normal can end at/before onset - CLEAN_END_MARGIN.

    是否能放下以 onset-3s 为终点上界的 clean-normal 负样本。
    """
    half = window_sec / 2.0
    span = (L - 1) * stride
    # first_center >= half, last_center = first + span <= onset - margin
    return onset - CLEAN_END_MARGIN >= half + span - 1e-9


def load_gt() -> list[dict]:
    rows = []
    with GT_CSV.open() as f:
        for r in csv.DictReader(f):
            vid = Path(r["path"]).stem
            rows.append(
                {
                    "video_id": vid,
                    "source_video_id": youtube_source_id(vid),
                    "path": f"test_videos/videos/{vid}.mp4",
                    "accident_time": float(r["accident_time"]),
                    "type": r.get("type", ""),
                    "scene_layout": r.get("scene_layout", ""),
                    "center_x": float(r["center_x"]) if r.get("center_x") else None,
                    "center_y": float(r["center_y"]) if r.get("center_y") else None,
                }
            )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", default=str(HERE / "data_accident_matched"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.10)
    p.add_argument("--fixed_len", type=int, default=8, help="L = #windows per clip")
    p.add_argument("--window_sec", type=float, default=2.0)
    p.add_argument("--stride_sec", type=float, default=0.25)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not PROBE.exists():
        raise SystemExit(f"Missing duration probe cache: {PROBE}")
    probe = json.loads(PROBE.read_text())

    clips = load_gt()
    for c in clips:
        info = probe[c["video_id"]]
        c["duration"] = float(info["duration"])
        assert abs(c["accident_time"] - float(info["onset"])) < 1e-3

    # Group by YouTube source, then assign whole groups to one split.
    # 按 YouTube 源分组，整组进入同一 split，防止同源泄漏。
    by_src: dict[str, list[dict]] = defaultdict(list)
    for c in clips:
        by_src[c["source_video_id"]].append(c)
    sources = sorted(by_src.keys())
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    n = len(sources)
    n_train = int(round(n * args.train_ratio))
    n_val = int(round(n * args.val_ratio))
    split_of = {}
    for i, sid in enumerate(sources):
        if i < n_train:
            split_of[sid] = "train"
        elif i < n_train + n_val:
            split_of[sid] = "val"
        else:
            split_of[sid] = "test"

    L = args.fixed_len
    matched = {"train": [], "val": [], "test": []}
    excluded = []
    all_rows = {"train": [], "val": [], "test": []}

    for c in clips:
        sid = c["source_video_id"]
        split = split_of[sid]
        onset, dur = c["accident_time"], c["duration"]
        pos_ok = can_place_positive(onset, dur, L, args.window_sec, args.stride_sec)
        neg_ok = can_place_negative(onset, L, args.window_sec, args.stride_sec)
        reasons = []
        if not pos_ok:
            reasons.append("no_positive_placement")
        if not neg_ok:
            reasons.append("no_clean_normal_negative")
        row = {
            **c,
            "split": split,
            "is_matched": bool(pos_ok and neg_ok),
            "prelim_pos_ok": pos_ok,
            "prelim_neg_ok": neg_ok,
            "exclude_reasons": reasons,
            "fixed_len": L,
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "clean_end_margin": CLEAN_END_MARGIN,
        }
        all_rows[split].append(row)
        if row["is_matched"]:
            matched[split].append(row)
        else:
            excluded.append(row)

    def write_jsonl(path: Path, rows: list[dict]):
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    for split in ("train", "val", "test"):
        write_jsonl(out / f"{split}_all.jsonl", all_rows[split])
        write_jsonl(out / f"{split}_matched.jsonl", matched[split])
        # extract_embeddings.py expects {split}_crash.jsonl
        crash_rows = []
        for r in matched[split]:
            crash_rows.append(
                {
                    "video_id": r["video_id"],
                    "path": r["path"],
                    "is_crash": 1,
                    "source": "accident",
                    "accident_time": r["accident_time"],
                    "center_x": r["center_x"],
                    "center_y": r["center_y"],
                    "type": r["type"],
                    "scene_layout": r["scene_layout"],
                    "source_video_id": r["source_video_id"],
                    "duration": r["duration"],
                }
            )
        write_jsonl(out / f"{split}_crash.jsonl", crash_rows)
        # empty non file so extract_embeddings --crash_only is optional
        write_jsonl(out / f"{split}_non.jsonl", [])

    write_jsonl(out / "excluded.jsonl", excluded)
    (out / "source_split.json").write_text(json.dumps(split_of, indent=2, sort_keys=True))

    meta = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "fixed_len": L,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "clean_end_margin": CLEAN_END_MARGIN,
        "phase_boundaries_sec": {
            "normal": "center < onset - 2.0",
            "pre_crash": "[onset - 2.0, onset - 0.5)",
            "impact": "abs(center - onset) <= 0.5",
            "post_crash": "center > onset + 0.5",
        },
        "n_source_youtube": len(sources),
        "n_clips_total": len(clips),
        "n_matched_prelim": {s: len(matched[s]) for s in matched},
        "n_excluded_prelim": len(excluded),
        "n_split_sources": {
            s: sum(1 for v in split_of.values() if v == s) for s in ("train", "val", "test")
        },
        "exclude_reason_counts": {},
        "note": (
            "Prelim eligibility uses continuous duration; finalize with "
            "0726_build_accident_matched_index.py after embedding extraction."
        ),
    }
    reason_counts: dict[str, int] = defaultdict(int)
    for r in excluded:
        for reason in r["exclude_reasons"] or ["unknown"]:
            reason_counts[reason] += 1
    meta["exclude_reason_counts"] = dict(reason_counts)
    (out / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
