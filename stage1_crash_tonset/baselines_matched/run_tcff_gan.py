#!/usr/bin/env python3
"""TCFF-GAN / SO-TAD on ACCIDENT matched clips.

VAE + temporal GAN trained on negative-clip frames. Clip score = max cosine
distance between predicted next latent and actual (higher = more anomalous).
在负样本 clip 上训练 VAE+GAN；分数为预测下一帧 latent 与真实值的最大余弦距离。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Vendored TCFF-GAN / SO-TAD sources shipped with this package.
# 随代码包附带的 TCFF-GAN / SO-TAD 源码。
SO_TAD = ROOT.parent / "third_party" / "so-tad"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SO_TAD))

from gan import Generator  # noqa: E402
from res_vae import VAE  # noqa: E402
import helper as hf  # noqa: E402
import loss_F  # noqa: E402

from clip_io import ClipSpec, load_catalog, load_clip_frames_ffmpeg  # noqa: E402
from metrics_matched import save_run  # noqa: E402

OUT_DEFAULT = ROOT / "results/accident_matched_baselines/tcff_gan"
IMG_SIZE = 64
FRAME_SPACE = 5
N_FRAMES = 4


def _tf():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def frame_to_tensor(rgb: np.ndarray, tf) -> torch.Tensor:
    return tf(Image.fromarray(rgb))


class VAEFrameDataset(Dataset):
    def __init__(self, specs: list[ClipSpec], frames_per_clip: int = 8):
        self.tf = _tf()
        self.frames: list[np.ndarray] = []
        for s in tqdm(specs, desc="vae cache neg clips"):
            for fr in load_clip_frames_ffmpeg(s, frames_per_clip, width=IMG_SIZE, height=IMG_SIZE):
                self.frames.append(fr)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return frame_to_tensor(self.frames[idx], self.tf)


class GANClipDataset(Dataset):
    """4-frame tuples sampled inside the matched window."""

    def __init__(self, specs: list[ClipSpec], samples_per_clip: int = 4):
        self.tf = _tf()
        self.items: list[list[np.ndarray]] = []
        need = N_FRAMES
        for s in tqdm(specs, desc="gan cache neg clips"):
            frs = load_clip_frames_ffmpeg(s, max(8, need * 2), width=IMG_SIZE, height=IMG_SIZE)
            if len(frs) < need:
                continue
            step = max(1, (len(frs) - need) // max(1, samples_per_clip))
            for off in range(0, len(frs) - need + 1, step):
                self.items.append(frs[off : off + need])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        frames = [frame_to_tensor(fr, self.tf).unsqueeze(1) for fr in self.items[idx]]
        return torch.cat(frames, dim=1)  # C,T,H,W


def train_vae(specs, device, epochs, batch_size, out_dir):
    ds = VAEFrameDataset(specs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    vae = VAE(channel_in=3, ch=64, latent_channels=256).to(device)
    opt = optim.Adam(vae.parameters(), lr=1e-4)
    t0 = time.time()
    for ep in range(epochs):
        vae.train()
        total = 0.0
        for images in tqdm(loader, desc=f"vae {ep+1}/{epochs}", leave=False):
            images = images.to(device)
            recon, mu, logvar = vae(images)
            loss = F.mse_loss(recon, images) + hf.kl_loss(mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        elapsed = time.time() - t0
        eta = elapsed / (ep + 1) * (epochs - ep - 1)
        print(
            f"VAE epoch {ep+1}/{epochs} loss={total/max(1,len(loader)):.5f} "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )
    torch.save(vae.state_dict(), out_dir / "vae.pt")
    return vae


def train_gan(specs, vae, device, epochs, batch_size, out_dir):
    ds = GANClipDataset(specs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    netG = Generator(16, 64, 16).to(device)
    netG.apply(hf.weights_init)
    optG = optim.Adam(netG.parameters(), lr=2e-3, betas=(0.5, 0.999))
    vae.eval()
    t0 = time.time()
    for ep in range(epochs):
        netG.train()
        last_g = 0.0
        for data in tqdm(loader, desc=f"gan {ep+1}/{epochs}", leave=False):
            real = data.to(device)
            t0f = torch.squeeze(real[:, :, 0:1, :, :], 2)
            t1f = torch.squeeze(real[:, :, 1:2, :, :], 2)
            t2f = torch.squeeze(real[:, :, 2:3, :, :], 2)
            t3f = torch.squeeze(real[:, :, -1:, :, :], 2)
            with torch.no_grad():
                mu0 = vae(t0f)[1]
                mu1 = vae(t1f)[1]
                mu2 = vae(t2f)[1]
                mu3 = vae(t3f)[1]
                d01 = F.normalize(mu1 - mu0, dim=1)
                d12 = F.normalize(mu2 - mu1, dim=1)
            noise = torch.randn_like(mu0)
            fake = netG(mu0 + torch.randn_like(mu0), mu1 + torch.randn_like(mu0), mu2 + noise, d01, d12)
            netG.zero_grad()
            errG = loss_F.cosine_similarity(fake, mu3)
            errG.backward()
            optG.step()
            last_g = float(errG.item())
        elapsed = time.time() - t0
        eta = elapsed / (ep + 1) * (epochs - ep - 1)
        print(
            f"GAN epoch {ep+1}/{epochs} G={last_g:.4f} "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )
    torch.save(netG.state_dict(), out_dir / "gan.pt")
    return netG


@torch.no_grad()
def score_clip(vae, gan, spec: ClipSpec, device) -> float:
    vae.eval()
    gan.eval()
    tf = _tf()
    frs = load_clip_frames_ffmpeg(spec, max(8, N_FRAMES * 3), width=IMG_SIZE, height=IMG_SIZE)
    if len(frs) < N_FRAMES:
        return 0.0
    scores = []
    for off in range(0, len(frs) - N_FRAMES + 1, 2):
        tensors = [frame_to_tensor(frs[off + k], tf) for k in range(N_FRAMES)]
        batch = torch.stack(tensors, dim=1).unsqueeze(0).to(device)  # 1,C,T,H,W
        t0f = torch.squeeze(batch[:, :, 0:1], 2)
        t1f = torch.squeeze(batch[:, :, 1:2], 2)
        t2f = torch.squeeze(batch[:, :, 2:3], 2)
        t3f = torch.squeeze(batch[:, :, -1:], 2)
        mu0 = vae(t0f)[1]
        mu1 = vae(t1f)[1]
        mu2 = vae(t2f)[1]
        mu3 = vae(t3f)[1]
        d01 = F.normalize(mu1 - mu0, dim=1)
        d12 = F.normalize(mu2 - mu1, dim=1)
        fake = gan(mu0, mu1, mu2, d01, d12)
        cos = 1 - F.cosine_similarity(fake.view(1, -1), mu3.view(1, -1), dim=1)
        scores.append(float(cos.item()))
    return float(max(scores)) if scores else 0.0


def score_specs(vae, gan, specs, device) -> list[dict]:
    rows = []
    for s in tqdm(specs, desc="tcff-gan score"):
        rows.append(
            {
                "clip_id": s.clip_id,
                "video_id": s.video_id,
                "role": s.role,
                "is_crash": int(s.is_crash),
                "score": score_clip(vae, gan, s, device),
            }
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vae_epochs", type=int, default=5)
    p.add_argument("--gan_epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--output_dir", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()

    train_neg = [s for s in load_catalog("train") if s.role == "negative"]
    val, test = load_catalog("val"), load_catalog("test")
    print(f"TCFF-GAN train-neg={len(train_neg)} val={len(val)} test={len(test)}", flush=True)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_path = out / "vae.pt"
    if vae_path.is_file():
        print(f"Loading existing VAE {vae_path}", flush=True)
        vae = VAE(channel_in=3, ch=64, latent_channels=256).to(device)
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        vae.eval()
    else:
        vae = train_vae(train_neg, device, args.vae_epochs, args.batch_size, out)
    gan = train_gan(train_neg, vae, device, args.gan_epochs, args.batch_size, out)
    save_run(
        out,
        method="TCFF-GAN / SO-TAD",
        val_rows=score_specs(vae, gan, val, device),
        test_rows=score_specs(vae, gan, test, device),
        extra={"vae_epochs": args.vae_epochs, "gan_epochs": args.gan_epochs, "img_size": IMG_SIZE},
    )


if __name__ == "__main__":
    main()
