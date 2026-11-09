#!/usr/bin/env python3
"""Extract frozen VideoMAE window embeddings + cumulative targets per video.

Saves: cache_emb/{split}/{video_id}.npz
  times, emb, y_pre, y_post, is_crash, t_gt, path

Usage
-----
  python 0722_cumcrash/extract_embeddings.py --split val
  python 0722_cumcrash/extract_embeddings.py --split train --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import VideoMAEImageProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from crashalign_paths import accident_root  # noqa: E402

# Videos and the VideoMAE checkpoint live in the data tree, not this package.
# 视频和 VideoMAE 权重在数据目录，不在本代码包内。
ROOT = accident_root()
from model import FrozenVideoMAE  # noqa: E402
from targets import cumulative_targets, window_specs  # noqa: E402

SPLIT_DIR = HERE / "data_split"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_all_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 25.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


@torch.no_grad()
def encode_video(encoder, processor, frames, fps, device, batch_size, window_sec, stride_sec):
    specs = window_specs(len(frames), fps, window_sec, stride_sec)
    if not specs:
        return None
    times = np.array([t for _, _, t in specs], dtype=np.float32)
    embs = []
    for i in range(0, len(specs), batch_size):
        chunk = specs[i : i + batch_size]
        tensors = []
        for s, e, _ in chunk:
            idx = np.linspace(s, max(s, e - 1), 16).astype(int)
            clip = [frames[j] for j in idx]
            tensors.append(processor(clip, return_tensors="pt")["pixel_values"].squeeze(0))
        pv = torch.stack(tensors).to(device)
        embs.append(encoder(pv).cpu().float().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    return times, emb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "val", "test"), required=True)
    p.add_argument("--pretrained", default=str(ROOT / "vit/checkpoints_temporal_clean/best_model"))
    p.add_argument("--output_dir", default=str(HERE / "cache_emb"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--window_sec", type=float, default=2.0)
    p.add_argument("--stride_sec", type=float, default=0.25)
    p.add_argument("--pre_lead", type=float, default=3.0)
    p.add_argument("--post_lag", type=float, default=0.5)
    p.add_argument("--post_end", type=float, default=4.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--include_non", action="store_true", default=True)
    p.add_argument("--crash_only", action="store_true")
    p.add_argument("--non_only", action="store_true")
    p.add_argument("--skip_existing", action="store_true", default=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenVideoMAE(args.pretrained).to(device)
    processor = VideoMAEImageProcessor.from_pretrained(args.pretrained)

    metas = []
    if not args.non_only:
        metas += load_jsonl(SPLIT_DIR / f"{args.split}_crash.jsonl")
    if args.include_non and not args.crash_only:
        metas += load_jsonl(SPLIT_DIR / f"{args.split}_non.jsonl")
    if args.limit:
        metas = metas[: args.limit]

    n_ok = n_skip = n_fail = 0
    for meta in tqdm(metas, desc=f"extract-{args.split}"):
        vid = str(meta["video_id"])
        out_path = out_dir / f"{vid}.npz"
        if args.skip_existing and out_path.exists():
            n_skip += 1
            continue
        frames, fps = load_all_frames(ROOT / meta["path"])
        if len(frames) < 2:
            n_fail += 1
            continue
        packed = encode_video(
            encoder,
            processor,
            frames,
            fps,
            device,
            args.batch_size,
            args.window_sec,
            args.stride_sec,
        )
        if packed is None:
            n_fail += 1
            continue
        times, emb = packed
        is_crash = int(meta["is_crash"])
        t_gt = float(meta["accident_time"]) if is_crash and meta.get("accident_time") is not None else None
        y_pre, y_post = cumulative_targets(
            times, t_gt, args.pre_lead, args.post_lag, args.post_end
        )
        np.savez_compressed(
            out_path,
            times=times,
            emb=emb,
            y_pre=y_pre,
            y_post=y_post,
            is_crash=np.array(is_crash, dtype=np.int64),
            t_gt=np.array(-1.0 if t_gt is None else t_gt, dtype=np.float32),
            path=np.array(meta["path"]),
            video_id=np.array(vid),
        )
        n_ok += 1

    manifest = {
        "split": args.split,
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_fail": n_fail,
        "n_listed": len(metas),
        "pretrained": args.pretrained,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "pre_lead": args.pre_lead,
        "post_lag": args.post_lag,
        "post_end": args.post_end,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
