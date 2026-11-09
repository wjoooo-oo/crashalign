#!/usr/bin/env python3
"""Full VideoMAE training on ACCIDENT matched clip windows.

Same recipe as vit/train_ensemble.py (videomae-base, MSE on Gaussian onset
targets, full unfreeze), but windows come only from matched pos/neg clips.
Clip-level TAD score = max window probability.

按原 VideoMAE 配方在 ACCIDENT matched clip 的 8 个 2s 窗口上全量训练；
clip 分数取窗口最大概率，用于 crash / non-crash。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "baselines_matched"))
from clip_io import FIXED_LEN, WINDOW_SEC, ClipSpec, load_catalog, load_clip_frames_ffmpeg  # noqa: E402
from metrics_matched import best_balanced_threshold, crash_metrics, save_run  # noqa: E402

VIDEOMAE_BASE = "MCG-NJU/videomae-base"
DEFAULT_OUT = HERE / "checkpoints/accident_matched/videomae_crash"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
FRAMES_PER_WINDOW = 16
STRIDE_FRAMES = 2  # 0.25s at 8 fps
CLIP_FRAMES = FRAMES_PER_WINDOW + (FIXED_LEN - 1) * STRIDE_FRAMES  # 30


def gaussian_target(t: float, t_gt: float, sigma: float = 2.0) -> float:
    return math.exp(-((t - t_gt) ** 2) / (2.0 * sigma * sigma))


def window_centers(spec: ClipSpec) -> list[float]:
    half = WINDOW_SEC / 2.0
    stride = (spec.t_end - spec.t_start - WINDOW_SEC) / max(1, FIXED_LEN - 1)
    return [spec.t_start + half + i * stride for i in range(FIXED_LEN)]


def frames_to_pixel_values(frames: list[np.ndarray]) -> torch.Tensor:
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # T,H,W,C
    arr = np.transpose(arr, (0, 3, 1, 2))  # T,C,H,W
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(arr))


def split_clip_windows(frames: list[np.ndarray]) -> list[list[np.ndarray]]:
    if not frames:
        blank = np.zeros((224, 224, 3), dtype=np.uint8)
        frames = [blank] * CLIP_FRAMES
    while len(frames) < CLIP_FRAMES:
        frames.append(frames[-1].copy())
    frames = frames[:CLIP_FRAMES]
    windows = []
    for i in range(FIXED_LEN):
        sl = frames[i * STRIDE_FRAMES : i * STRIDE_FRAMES + FRAMES_PER_WINDOW]
        if len(sl) < FRAMES_PER_WINDOW:
            sl = sl + [sl[-1].copy()] * (FRAMES_PER_WINDOW - len(sl))
        windows.append(sl[:FRAMES_PER_WINDOW])
    return windows


class MatchedClipWindowDataset(Dataset):
    def __init__(self, specs: list[ClipSpec], *, train: bool = False, sigma: float = 2.0, seed: int = 42):
        self.specs = specs
        self.train = train
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx: int):
        spec = self.specs[idx]
        frames = load_clip_frames_ffmpeg(spec, CLIP_FRAMES, width=224, height=224)
        windows = split_clip_windows(frames)
        if self.train and self.rng.random() < 0.5:
            windows = [[np.ascontiguousarray(np.flip(fr, axis=1)) for fr in w] for w in windows]
        pixel_values = torch.stack([frames_to_pixel_values(w) for w in windows], 0)
        centers = window_centers(spec)
        if spec.is_crash:
            targets = [gaussian_target(c, spec.onset, self.sigma) for c in centers]
        else:
            targets = [0.0] * FIXED_LEN
        return {
            "pixel_values": pixel_values,
            "targets": torch.tensor(targets, dtype=torch.float32),
            "is_crash": torch.tensor(float(spec.is_crash), dtype=torch.float32),
            "clip_id": spec.clip_id,
            "video_id": spec.video_id,
            "role": spec.role,
        }


def collate(batch):
    return {
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], 0),
        "targets": torch.cat([b["targets"] for b in batch], 0),
        "is_crash": torch.stack([b["is_crash"] for b in batch], 0),
        "clip_id": [b["clip_id"] for b in batch],
        "video_id": [b["video_id"] for b in batch],
        "role": [b["role"] for b in batch],
        "n_windows": FIXED_LEN,
    }


@torch.no_grad()
def score_loader(model, loader, device) -> tuple[list[dict], float]:
    model.eval()
    rows = []
    total_loss = 0.0
    n_batches = 0
    loss_fn = nn.MSELoss()
    for batch in loader:
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["targets"].to(device, non_blocking=True)
        with autocast(enabled=device.type == "cuda"):
            logits = model(pixel_values=x).logits.squeeze(-1)
            preds = torch.sigmoid(logits.float())
            loss = loss_fn(preds, y)
        total_loss += float(loss.item())
        n_batches += 1
        n_w = batch["n_windows"]
        probs = preds.view(-1, n_w).cpu().numpy()
        clip_scores = probs.max(axis=1)
        labels = batch["is_crash"].cpu().numpy()
        for i in range(len(batch["clip_id"])):
            rows.append(
                {
                    "clip_id": batch["clip_id"][i],
                    "video_id": batch["video_id"][i],
                    "role": batch["role"][i],
                    "is_crash": int(labels[i] > 0.5),
                    "score": float(clip_scores[i]),
                }
            )
    return rows, total_loss / max(1, n_batches)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init_ckpt", type=str, default=VIDEOMAE_BASE)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=2, help="clips per step; each clip is 8 windows")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--sigma", type=float, default=2.0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad_clip", type=float, default=1.0)
    args = p.parse_args()

    set_seed(args.seed)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} init={args.init_ckpt}", flush=True)

    processor = VideoMAEImageProcessor.from_pretrained(args.init_ckpt)
    model = VideoMAEForVideoClassification.from_pretrained(args.init_ckpt, num_labels=1)
    model = model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {n_train:,} / {n_all:,} ({100.0 * n_train / n_all:.2f}%)", flush=True)

    train_specs = load_catalog("train")
    val_specs = load_catalog("val")
    test_specs = load_catalog("test")
    print(
        f"clips train/val/test={len(train_specs)}/{len(val_specs)}/{len(test_specs)} "
        f"windows/clip={FIXED_LEN}",
        flush=True,
    )

    train_loader = DataLoader(
        MatchedClipWindowDataset(train_specs, train=True, sigma=args.sigma, seed=args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        MatchedClipWindowDataset(val_specs, train=False, sigma=args.sigma, seed=args.seed),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        MatchedClipWindowDataset(test_specs, train=False, sigma=args.sigma, seed=args.seed),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
        persistent_workers=args.num_workers > 0,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.MSELoss()

    (out / "config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "init_ckpt": str(args.init_ckpt),
                "output_dir": str(out),
                "protocol": "ACCIDENT matched window-level VideoMAE (train_ensemble recipe)",
                "n_train_clips": len(train_specs),
                "n_val_clips": len(val_specs),
                "n_test_clips": len(test_specs),
                "windows_per_clip": FIXED_LEN,
                "clip_score": "max window probability",
            },
            indent=2,
            default=str,
        )
    )

    history = []
    best_ba = -1.0
    bad = 0
    t0 = time.time()
    log_path = out / "train.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    for ep in range(args.epochs):
        model.train()
        running = 0.0
        n_b = 0
        pbar = tqdm(train_loader, desc=f"epoch {ep+1}/{args.epochs}", leave=False)
        for batch in pbar:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["targets"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(pixel_values=x).logits.squeeze(-1)
                preds = torch.sigmoid(logits.float())
                loss = loss_fn(preds, y)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss.item())
            n_b += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()

        val_rows, val_loss = score_loader(model, val_loader, device)
        yv = [r["is_crash"] for r in val_rows]
        sv = [r["score"] for r in val_rows]
        thr, _ = best_balanced_threshold(yv, sv)
        vm = crash_metrics(yv, sv, thr)
        elapsed = time.time() - t0
        eta = elapsed / (ep + 1) * (args.epochs - ep - 1)
        rec = {
            "epoch": ep + 1,
            "train_loss": running / max(1, n_b),
            "val_loss": val_loss,
            "val_ba": vm["balanced_accuracy"],
            "val_auroc": vm["auroc"],
            "val_f1": vm["f1"],
            "thr": thr,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        log(
            f"[epoch {ep+1}/{args.epochs}] train_loss={rec['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} val_BA={vm['balanced_accuracy']:.4f} "
            f"AUROC={vm['auroc']:.4f} F1={vm['f1']:.4f} thr={thr:.3f} "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m"
        )

        if vm["balanced_accuracy"] > best_ba + 1e-4:
            best_ba = vm["balanced_accuracy"]
            bad = 0
            best_dir = out / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_dir)
            processor.save_pretrained(best_dir)
            (out / "best_threshold.json").write_text(
                json.dumps({"best_thr": thr, "val_ba": best_ba, "epoch": ep + 1}, indent=2)
            )
            log(f"  saved best_model val_BA={best_ba:.4f}")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop at epoch {ep+1} (patience {args.patience})")
                break

    best_dir = out / "best_model"
    eval_model = VideoMAEForVideoClassification.from_pretrained(str(best_dir), num_labels=1).to(device)
    val_rows, _ = score_loader(eval_model, val_loader, device)
    test_rows, _ = score_loader(eval_model, test_loader, device)
    results_dir = HERE / "results/accident_matched_baselines/videomae_accident"
    payload = save_run(
        results_dir,
        method="VideoMAE (ACCIDENT clips)",
        val_rows=val_rows,
        test_rows=test_rows,
        extra={"init": str(args.init_ckpt), "checkpoint": str(best_dir), "clip_score": "max window prob"},
    )
    (out / "test_metrics.json").write_text(json.dumps(payload, indent=2))
    log(
        f"TEST BA={payload['test']['balanced_accuracy']:.4f} "
        f"AUROC={payload['test']['auroc']:.4f} F1={payload['test']['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
