#!/usr/bin/env python3
"""MDTS-ADNet dual-stream ConvLSTM VAE on ACCIDENT matched clips.

Train on negative clips only. Clip score = mean reconstruction MSE
(higher = more anomalous). Do not min-max within a clip — that collapses ranking.
仅在负样本 clip 上训练；分数用平均重建 MSE，不用 clip 内 min-max。
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXP = ROOT.parent.parent / "experiments_aap" / "sotad_tad"
MDTS_ROOT = EXP / "baselines/mdts_adnet/MDTS-ADNet"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(MDTS_ROOT))

from model.MDTS_ADNet import TwoStreamModel  # noqa: E402
from utils import loss_function  # noqa: E402
from src.video_features import optical_flow_mag  # noqa: E402

from clip_io import ClipSpec, load_catalog, load_clip_frames_ffmpeg  # noqa: E402
from metrics_matched import save_run  # noqa: E402

OUT_DEFAULT = ROOT / "results/accident_matched_baselines/mdts_adnet"


def mdts_norm_rgb(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (rgb / 127.5) - 1.0


def mdts_norm_flow(prev_bgr: np.ndarray, curr_bgr: np.ndarray, size: int) -> np.ndarray:
    prev = cv2.resize(prev_bgr, (size, size), interpolation=cv2.INTER_AREA)
    curr = cv2.resize(curr_bgr, (size, size), interpolation=cv2.INTER_AREA)
    prev_g = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_g = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    mag = optical_flow_mag(prev_g, curr_g)
    mag = cv2.resize(mag, (size, size), interpolation=cv2.INTER_AREA)
    mag = mag / (mag.max() + 1e-6)
    flow = np.stack([mag, mag, mag], axis=-1).astype(np.float32)
    return (flow * 2.0) - 1.0


def stack_pair(f0: np.ndarray, f1: np.ndarray) -> torch.Tensor:
    x = np.concatenate([f0.transpose(2, 0, 1), f1.transpose(2, 0, 1)], axis=0)
    return torch.from_numpy(x).float()


class PairClipDataset(Dataset):
    def __init__(self, specs: list[ClipSpec], *, size: int, pairs_per_clip: int = 8, n_sample: int = 16):
        self.size = size
        self.cache: dict[str, list[np.ndarray]] = {}
        self.samples: list[tuple[str, int]] = []
        for s in tqdm(specs, desc="mdts cache neg clips"):
            frames = load_clip_frames_ffmpeg(s, n_sample, width=size, height=size)
            if len(frames) < 2:
                continue
            self.cache[s.clip_id] = frames
            for _ in range(pairs_per_clip):
                self.samples.append((s.clip_id, random.randint(0, len(frames) - 2)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cid, off = self.samples[idx]
        frames = self.cache[cid]
        f0, f1 = frames[off], frames[off + 1]
        b0 = cv2.cvtColor(f0, cv2.COLOR_RGB2BGR)
        b1 = cv2.cvtColor(f1, cv2.COLOR_RGB2BGR)
        return stack_pair(mdts_norm_rgb(b0, self.size), mdts_norm_rgb(b1, self.size)), stack_pair(
            mdts_norm_flow(b0, b0, self.size), mdts_norm_flow(b0, b1, self.size)
        )


def reset_hidden(model: TwoStreamModel, batch: int, hw: int) -> None:
    model.reset_hidden(batch, (hw, hw))


def train_model(model, loader, device, epochs: int, lr: float) -> None:
    params = (
        list(model.rgb_model.encoder.parameters())
        + list(model.rgb_model.decoder.parameters())
        + list(model.of_model.encoder.parameters())
        + list(model.of_model.decoder.parameters())
        + [model.weight_rgb]
    )
    opt = optim.Adam(params, lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    model.train()
    t0 = time.time()
    for ep in range(epochs):
        total = 0.0
        for imgs_rgb, imgs_of in tqdm(loader, desc=f"mdts ep{ep+1}/{epochs}", leave=False):
            imgs_rgb, imgs_of = imgs_rgb.to(device), imgs_of.to(device)
            reset_hidden(model, imgs_rgb.size(0), imgs_rgb.size(2))
            _c, rgb_out, of_out, rgb_mu, rgb_lv, of_mu, of_lv, _w, w_rgb = model(imgs_rgb, imgs_of)
            opt.zero_grad()
            loss = loss_function(rgb_out, of_out, imgs_rgb, imgs_of, rgb_mu, rgb_lv, of_mu, of_lv, w_rgb)
            loss.backward()
            opt.step()
            total += float(loss.item())
        sched.step()
        elapsed = time.time() - t0
        eta = elapsed / (ep + 1) * (epochs - ep - 1)
        print(
            f"epoch {ep+1}/{epochs} loss={total/max(1,len(loader)):.6f} "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )


@torch.no_grad()
def score_clip(model, spec: ClipSpec, device, size: int, n_sample: int = 16) -> float:
    """Mean RGB+flow recon MSE over sampled pairs inside the clip only."""
    model.eval()
    frames = load_clip_frames_ffmpeg(spec, n_sample, width=size, height=size)
    if len(frames) < 2:
        return 0.0
    reset_hidden(model, 1, size)
    mses = []
    prev = frames[0]
    for curr in frames[1:]:
        b0 = cv2.cvtColor(prev, cv2.COLOR_RGB2BGR)
        b1 = cv2.cvtColor(curr, cv2.COLOR_RGB2BGR)
        imgs_rgb = stack_pair(mdts_norm_rgb(b0, size), mdts_norm_rgb(b1, size)).unsqueeze(0).to(device)
        imgs_of = stack_pair(mdts_norm_flow(b0, b0, size), mdts_norm_flow(b0, b1, size)).unsqueeze(0).to(device)
        _c, rgb_out, of_out, *_ = model(imgs_rgb, imgs_of)
        mse_rgb = float(F.mse_loss((rgb_out[0] + 1) / 2, (imgs_rgb[0] + 1) / 2).item())
        mse_of = float(F.mse_loss((of_out[0] + 1) / 2, (imgs_of[0] + 1) / 2).item())
        mses.append(0.5 * (mse_rgb + mse_of))
        prev = curr
    return float(np.mean(mses)) if mses else 0.0


def score_specs(model, specs, device, size) -> list[dict]:
    rows = []
    for s in tqdm(specs, desc="mdts score"):
        rows.append(
            {
                "clip_id": s.clip_id,
                "video_id": s.video_id,
                "role": s.role,
                "is_crash": int(s.is_crash),
                "score": score_clip(model, s, device, size),
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--pairs_per_clip", type=int, default=8)
    p.add_argument("--output_dir", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()

    train_neg = [s for s in load_catalog("train") if s.role == "negative"]
    val, test = load_catalog("val"), load_catalog("test")
    print(f"MDTS train-neg={len(train_neg)} val={len(val)} test={len(test)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoStreamModel(rgb_channels=3, of_channels=3, t_length=2, latent_dim=128).to(device)
    ds = PairClipDataset(train_neg, size=args.size, pairs_per_clip=args.pairs_per_clip)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    train_model(model, loader, device, args.epochs, args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output_dir / "model.pt")
    save_run(
        args.output_dir,
        method="MDTS-ADNet",
        val_rows=score_specs(model, val, device, args.size),
        test_rows=score_specs(model, test, device, args.size),
        extra={"epochs": args.epochs, "train_neg_only": True, "score": "mean_recon_mse"},
    )


if __name__ == "__main__":
    main()
