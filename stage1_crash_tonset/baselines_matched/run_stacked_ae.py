#!/usr/bin/env python3
"""Stacked Autoencoder trained on matched non-crash clips; score = max recon MSE."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clip_io import ClipSpec, load_catalog, load_clip_frames_rgb  # noqa: E402
from metrics_matched import save_run  # noqa: E402

ROOT = HERE.parent
OUT_DEFAULT = ROOT / "results/accident_matched_baselines/stacked_autoencoder"
IM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(frame: np.ndarray, size: int) -> np.ndarray:
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - IM_MEAN) / IM_STD
    return x.transpose(2, 0, 1)


def denorm(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x * std + mean).clamp(0, 1)


class StackedAE(nn.Module):
    def __init__(self, size: int = 128):
        super().__init__()
        flat = 3 * size * size
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Linear(512, flat),
            nn.Sigmoid(),
        )
        self.size = size

    def forward(self, x):
        b = x.size(0)
        return self.decoder(self.encoder(x)).view(b, 3, self.size, self.size)


class FrameDataset(Dataset):
    def __init__(self, specs: list[ClipSpec], size: int, n_frames: int):
        self.specs = specs
        self.size = size
        self.n_frames = n_frames

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        frames = load_clip_frames_rgb(self.specs[idx], self.n_frames)
        if not frames:
            x = np.zeros((self.n_frames, 3, self.size, self.size), dtype=np.float32)
        else:
            xs = [preprocess(f, self.size) for f in frames]
            while len(xs) < self.n_frames:
                xs.append(xs[-1])
            x = np.stack(xs[: self.n_frames]).astype(np.float32)
        return torch.from_numpy(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--n_frames", type=int, default=8)
    p.add_argument("--output_dir", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()

    t0 = time.time()
    train = [s for s in load_catalog("train") if s.role == "negative"]
    val, test = load_catalog("val"), load_catalog("test")
    print(f"AE train-on-neg={len(train)} val={len(val)} test={len(test)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StackedAE(size=args.size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        FrameDataset(train, args.size, args.n_frames),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    model.train()
    n_ep = args.epochs
    for ep in range(n_ep):
        total = 0.0
        for x in tqdm(loader, desc=f"stacked-ae ep{ep+1}/{n_ep}", leave=False):
            x = denorm(x.to(device).view(-1, 3, args.size, args.size))
            opt.zero_grad()
            loss = loss_fn(model(x), x)
            loss.backward()
            opt.step()
            total += float(loss.item())
        elapsed = time.time() - t0
        eta = elapsed / (ep + 1) * (n_ep - ep - 1)
        print(
            f"epoch {ep+1}/{n_ep} loss={total/max(1,len(loader)):.5f} "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )

    @torch.no_grad()
    def score_specs(specs: list[ClipSpec]) -> list[dict]:
        model.eval()
        rows = []
        for s in tqdm(specs, desc="stacked-ae score"):
            frames = load_clip_frames_rgb(s, args.n_frames)
            errs = []
            for fr in frames:
                x = torch.from_numpy(preprocess(fr, args.size)).unsqueeze(0).to(device)
                x = denorm(x)
                recon = model(x)
                errs.append(float(torch.mean((recon - x) ** 2).item()))
            rows.append(
                {
                    "clip_id": s.clip_id,
                    "video_id": s.video_id,
                    "role": s.role,
                    "is_crash": int(s.is_crash),
                    "score": float(max(errs) if errs else 0.0),
                }
            )
        return rows

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    save_run(
        args.output_dir,
        method="Stacked Autoencoder",
        val_rows=score_specs(val),
        test_rows=score_specs(test),
        extra={"epochs": args.epochs, "train_neg_only": True},
    )


if __name__ == "__main__":
    main()
