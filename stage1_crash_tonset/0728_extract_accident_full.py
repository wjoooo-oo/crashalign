#!/usr/bin/env python3
# 0728_extract_accident_full.py
"""Cache frozen VideoMAE embeddings for the *complete* ACCIDENT dataset.

Why the full set / 为什么用全量:
  `cache_emb_accident` holds only the 721 videos of the matched detection
  protocol. Of the 1306 videos it left out, 1266 were dropped solely because the
  same video offers no clean-normal negative clip — a Task-1 *detection*
  requirement that is irrelevant to onset localization, which needs no
  negatives at all. So all 2027 annotated crash videos are usable for T.
  被排除的 1306 个视频中有 1266 个只是因为同一视频内找不到 clean-normal 负样本，
  那是检测任务的约束；onset 定位不需要负样本，故全部 2027 个视频均可用。

Grid matches the existing cache exactly (window 2.0 s, stride 0.25 s, same
frozen encoder), so newly extracted videos are interchangeable with the ones
already cached. Verified by `--verify`.
窗口网格与现有缓存完全一致，可与已有文件混用；用 --verify 校验。

Speed / 速度: two changes, both numerically neutral.
  1. Consecutive windows overlap by ~90 %, so every frame is preprocessed once
     and then gathered per window instead of being re-processed for each
     window. Identical results because the VideoMAE processor treats frames
     independently (resize, crop, rescale, normalize).
     相邻窗口重叠约 90%，每帧只预处理一次再按窗口取用；预处理逐帧独立，结果等价。
  2. Decoding and preprocessing run in DataLoader workers so they overlap with
     GPU inference; measured single-process runs left the GPU idle ~70 % of
     the time.
     解码与预处理放进 DataLoader worker，与 GPU 推理重叠；
     单进程实测 GPU 约 70% 时间空闲。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
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

# Full ACCIDENT videos and the frozen encoder live in the data tree.
# 全量视频和冻结编码器在数据目录。
ROOT = accident_root()
from labels import assign_phase, phase_to_cumulative, window_specs  # noqa: E402
from model import FrozenVideoMAE  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_all_frames(path: Path, max_sec: float | None = None):
    """Decode frames, optionally stopping after `max_sec` seconds.

    解码视频帧；给定 max_sec 时解码到该时刻即停止（截尾缓存无需整段解码）。
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return [], 25.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    max_frames = None if max_sec is None else int(max_sec * fps) + 2
    frames = []
    while max_frames is None or len(frames) < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def preprocess_video(processor, frames, fps, window_sec: float, stride_sec: float,
                     prep_chunk: int = 256, max_center: float | None = None):
    """CPU part: window grid + one-pass frame preprocessing.

    `max_center` drops the tail of the grid (windows centred after it); the kept
    windows are identical to the full grid, only fewer.
    CPU 部分：计算窗口网格，并对所有帧做一次性预处理。
    `max_center` 截掉网格尾部，保留的窗口与完整网格完全一致。
    """
    specs = window_specs(len(frames), fps, window_sec, stride_sec)
    if max_center is not None:
        specs = [s for s in specs if s[2] <= max_center]
    if not specs:
        return None
    times = np.array([t for _, _, t in specs], dtype=np.float32)
    prepped = []
    for i in range(0, len(frames), prep_chunk):
        pv = processor(list(frames[i : i + prep_chunk]), return_tensors="pt")
        prepped.append(pv["pixel_values"].squeeze(0))
    all_frames = torch.cat(prepped, dim=0)  # [N,3,224,224]
    gather = np.stack([np.linspace(s, max(s, e - 1), 16).astype(int) for s, e, _ in specs])
    return times, all_frames, torch.from_numpy(gather)


@torch.no_grad()
def encode_prepared(encoder, all_frames, gather, device, batch_size, amp: bool = False):
    """GPU part: gather 16-frame clips and run the frozen encoder.

    GPU 部分：按索引取 16 帧片段并送入冻结编码器；amp=True 时用 bf16 加速。
    """
    embs = []
    for i in range(0, gather.shape[0], batch_size):
        idx = gather[i : i + batch_size]
        clips = all_frames[idx.reshape(-1)]
        clips = clips.view(idx.shape[0], 16, *all_frames.shape[1:]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            embs.append(encoder(clips).cpu().float().numpy())
    return np.concatenate(embs, axis=0).astype(np.float32)


@torch.no_grad()
def encode_video(encoder, processor, frames, fps, device, batch_size,
                 window_sec: float, stride_sec: float):
    """Single-process path, kept for `--verify`. / 单进程路径，供 --verify 使用。"""
    packed = preprocess_video(processor, frames, fps, window_sec, stride_sec)
    if packed is None:
        return None
    times, all_frames, gather = packed
    return times, encode_prepared(encoder, all_frames, gather, device, batch_size)


class VideoPrepDataset(torch.utils.data.Dataset):
    """Decode + preprocess one video per item, inside a worker process.

    每个 item 在 worker 进程里完成一个视频的解码与预处理。
    """

    def __init__(self, rows: list[dict], processor, window_sec: float, stride_sec: float,
                 max_center_after_onset: float | None = None):
        self.rows = rows
        self.processor = processor
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.max_center_after_onset = max_center_after_onset

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        max_center = None
        if self.max_center_after_onset is not None:
            max_center = float(r["accident_time"]) + self.max_center_after_onset
        # Decode one extra window beyond the last kept centre so the grid is unchanged.
        # 多解码一个窗口长度，保证保留部分的网格不变。
        max_sec = None if max_center is None else max_center + self.window_sec
        frames, fps = load_all_frames(ROOT / r["path"], max_sec=max_sec)
        if len(frames) < 2:
            return {"row": r, "ok": False}
        packed = preprocess_video(
            self.processor, frames, fps, self.window_sec, self.stride_sec, max_center=max_center
        )
        if packed is None:
            return {"row": r, "ok": False}
        times, all_frames, gather = packed
        return {"row": r, "ok": True, "times": times, "frames": all_frames, "gather": gather}


def build_targets(times: np.ndarray, is_crash: int, t_gt: float | None):
    y_a = np.zeros(len(times), dtype=np.float32)
    y_b = np.zeros(len(times), dtype=np.float32)
    phase_valid = np.ones(len(times), dtype=np.bool_)
    for i, t in enumerate(times):
        ph = assign_phase(float(t), t_gt) if is_crash else "normal"
        ya, yb, valid = phase_to_cumulative(ph)
        phase_valid[i] = valid
        if valid:
            y_a[i] = float(ya)
            y_b[i] = float(yb)
    return y_a, y_b, phase_valid


def verify_against_cache(encoder, processor, device, args, n: int) -> None:
    """Re-extract already cached videos and compare, validating the fast path.

    重新提取已缓存的视频并比对，验证快速预处理路径的等价性。
    """
    ref_dir = Path(args.link_from) / "train"
    files = sorted(ref_dir.glob("*.npz"))[:n]
    print(f"Verifying fast path on {len(files)} cached videos")
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            ref_emb, ref_times, path = z["emb"], z["times"], str(z["path"])
        frames, fps = load_all_frames(ROOT / path)
        times, emb = encode_video(
            encoder, processor, frames, fps, device, args.batch_size,
            args.window_sec, args.stride_sec,
        )
        dt = float(np.abs(times - ref_times).max())
        de = float(np.abs(emb - ref_emb).max())
        rel = de / (float(np.abs(ref_emb).max()) + 1e-9)
        print(
            f"  {f.stem}: n={len(times)}/{len(ref_times)} "
            f"max|dt|={dt:.2e} max|demb|={de:.2e} rel={rel:.2e}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--data_dir", default=str(HERE / "data_accident_matched"))
    p.add_argument("--output_dir", default=str(HERE / "cache_emb_accident_full"))
    p.add_argument(
        "--link_from",
        default=str(HERE / "cache_emb_accident"),
        help="Existing cache to hardlink from / 已有缓存，直接硬链接复用",
    )
    p.add_argument("--pretrained", default=str(ROOT / "vit/checkpoints_temporal_clean/best_model"))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--num_workers", type=int, default=10,
        help="Worker processes for decode+preprocess / 解码与预处理的工作进程数",
    )
    p.add_argument("--window_sec", type=float, default=2.0)
    p.add_argument("--stride_sec", type=float, default=0.25)
    p.add_argument("--limit", type=int, default=None, help="Extract at most N per split")
    p.add_argument("--verify", type=int, default=0, help="Verify fast path on N cached videos")
    p.add_argument("--pooling", choices=("auto", "mean"), default="auto",
                   help="Token pooling, see FrozenVideoMAE / token 池化方式")
    p.add_argument("--video_ids_from", nargs="*", default=None,
                   help="Only extract videos listed in these jsonl files / 只提取这些清单中的视频")
    p.add_argument("--max_center_after_onset", type=float, default=None,
                   help="Drop windows centred later than onset + this (TAD-only caches) / 截掉 onset 之后的尾部窗口")
    p.add_argument("--amp", action="store_true", help="bf16 inference / bf16 推理")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = FrozenVideoMAE(args.pretrained, pooling=args.pooling).to(device)
    processor = VideoMAEImageProcessor.from_pretrained(args.pretrained)

    if args.verify:
        verify_against_cache(encoder, processor, device, args, args.verify)
        return

    data_dir = Path(args.data_dir)
    out_root = Path(args.output_dir)
    link_root = Path(args.link_from) if args.link_from else None
    keep_ids = None
    if args.video_ids_from:
        keep_ids = {str(r["video_id"]) for f in args.video_ids_from for r in load_jsonl(Path(f))}
    summary = {}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        rows = load_jsonl(data_dir / f"{split}_all.jsonl")
        if keep_ids is not None:
            rows = [r for r in rows if str(r["video_id"]) in keep_ids]
        out_dir = out_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        # Reuse anything already cached; only the rest costs GPU time.
        # 已缓存的直接硬链接复用，只对其余视频付出 GPU 代价。
        todo = []
        n_link = 0
        for r in rows:
            vid = str(r["video_id"])
            dst = out_dir / f"{vid}.npz"
            if dst.exists():
                n_link += 1
                continue
            src = link_root / split / f"{vid}.npz" if link_root else None
            if src is not None and src.exists():
                try:
                    dst.hardlink_to(src.resolve())
                except OSError:
                    dst.symlink_to(src.resolve())
                n_link += 1
                continue
            todo.append(r)
        if args.limit:
            todo = todo[: args.limit]
        print(f"[{split}] listed={len(rows)} reused={n_link} to_extract={len(todo)}")

        n_ok = n_fail = 0
        t0 = time.time()
        loader = torch.utils.data.DataLoader(
            VideoPrepDataset(todo, processor, args.window_sec, args.stride_sec,
                             args.max_center_after_onset),
            batch_size=None,  # one video per item / 每次一个视频
            shuffle=False,
            num_workers=args.num_workers,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )
        for item in tqdm(loader, total=len(todo), desc=f"extract-{split}"):
            r = item["row"]
            if not item["ok"]:
                n_fail += 1
                continue
            vid = str(r["video_id"])
            # DataLoader may have converted the array to a tensor / 可能已被转成张量
            times = np.asarray(item["times"], dtype=np.float32)
            emb = encode_prepared(
                encoder, item["frames"], item["gather"], device, args.batch_size, amp=args.amp
            )
            t_gt = r.get("accident_time")
            t_gt = float(t_gt) if t_gt is not None else None
            y_a, y_b, phase_valid = build_targets(times, 1, t_gt)
            np.savez_compressed(
                out_dir / f"{vid}.npz",
                times=times,
                emb=emb,
                y_a=y_a,
                y_b=y_b,
                phase_valid=phase_valid,
                is_crash=np.array(1, dtype=np.int64),
                t_gt=np.array(-1.0 if t_gt is None else t_gt, dtype=np.float32),
                path=np.array(r["path"]),
                video_id=np.array(vid),
            )
            n_ok += 1
        elapsed = time.time() - t0
        summary[split] = {
            "listed": len(rows),
            "reused": n_link,
            "extracted": n_ok,
            "failed": n_fail,
            "seconds": round(elapsed, 1),
            "sec_per_video": round(elapsed / max(n_ok, 1), 2),
        }
        print(f"[{split}] {summary[split]}")

    manifest = {
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "pretrained": args.pretrained,
        "pooling": args.pooling,
        "max_center_after_onset": args.max_center_after_onset,
        "amp": args.amp,
        "splits": summary,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
