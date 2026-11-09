#!/usr/bin/env python3
"""Cache frozen VideoMAE embeddings + cumulative phase targets per video."""

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
from labels import assign_phase, phase_to_cumulative, window_specs  # noqa: E402
from model import FrozenVideoMAE  # noqa: E402

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
            pv = processor(clip, return_tensors="pt")["pixel_values"].squeeze(0)
            tensors.append(pv)
        pv = torch.stack(tensors).to(device)
        embs.append(encoder(pv).cpu().float().numpy())
    emb = np.concatenate(embs, axis=0).astype(np.float32)
    return times, emb


def build_targets(times: np.ndarray, is_crash: int, t_gt: float | None):
    y_a = np.zeros(len(times), dtype=np.float32)
    y_b = np.zeros(len(times), dtype=np.float32)
    phase_valid = np.ones(len(times), dtype=np.bool_)
    phases = []
    counts = {"normal": 0, "pre_impact": 0, "impact": 0, "post_impact": 0}
    for i, t in enumerate(times):
        if not is_crash:
            ph = "normal"
        else:
            ph = assign_phase(float(t), t_gt)
        phases.append(ph)
        counts[ph] += 1
        ya, yb, valid = phase_to_cumulative(ph)
        phase_valid[i] = valid
        if valid:
            y_a[i] = float(ya)
            y_b[i] = float(yb)
    return y_a, y_b, phase_valid, phases, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "val", "test"), required=True)
    p.add_argument(
        "--data_dir",
        default=str(HERE / "data_split"),
        help="Directory with {split}_crash.jsonl and {split}_non.jsonl",
    )
    p.add_argument("--pretrained", default=str(ROOT / "vit/checkpoints_temporal_clean/best_model"))
    p.add_argument("--output_dir", default=str(HERE / "cache_emb"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--window_sec", type=float, default=2.0)
    p.add_argument("--stride_sec", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--crash_only", action="store_true")
    p.add_argument("--non_only", action="store_true")
    p.add_argument("--skip_existing", action="store_true", default=True)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"data_dir not found: {data_dir}")

    out_dir = Path(args.output_dir) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenVideoMAE(args.pretrained).to(device)
    processor = VideoMAEImageProcessor.from_pretrained(args.pretrained)

    metas = []
    if not args.non_only:
        metas += load_jsonl(data_dir / f"{args.split}_crash.jsonl")
    if not args.crash_only:
        metas += load_jsonl(data_dir / f"{args.split}_non.jsonl")
    if args.limit:
        metas = metas[: args.limit]

    tot_counts = {"normal": 0, "pre_impact": 0, "impact": 0, "post_impact": 0}
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
            encoder, processor, frames, fps, device, args.batch_size, args.window_sec, args.stride_sec
        )
        if packed is None:
            n_fail += 1
            continue
        times, emb = packed
        is_crash = int(meta["is_crash"])
        t_gt = float(meta["accident_time"]) if is_crash and meta.get("accident_time") is not None else None
        y_a, y_b, phase_valid, phases, counts = build_targets(times, is_crash, t_gt)
        for k, v in counts.items():
            tot_counts[k] += v
        np.savez_compressed(
            out_path,
            times=times,
            emb=emb,
            y_a=y_a,
            y_b=y_b,
            phase_valid=phase_valid,
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
        "phase_window_counts": tot_counts,
        "pretrained": args.pretrained,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "label_scheme": "normal(0,0) pre(1,0) impact(ignore) post(1,1) non(0,0)",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
