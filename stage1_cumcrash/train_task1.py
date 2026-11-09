#!/usr/bin/env python3
"""Train Task1 cumulative crash pattern on cached embeddings.

Loss = BCE(q_pre, y_pre) + BCE(q_post, y_post) + λ_crash * BCE(crash, y_crash)
Non-crash sequences are fully zero-risk (y_pre=y_post=0).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CumCrashModel  # noqa: E402


class SeqEmbDataset(Dataset):
    def __init__(self, cache_dir: Path, max_len: int = 256):
        self.files = sorted(cache_dir.glob("*.npz"))
        self.max_len = max_len
        if not self.files:
            raise FileNotFoundError(f"No .npz in {cache_dir}")

    def __len__(self):
        return len(self.files)

    def labels(self) -> list[int]:
        out = []
        for f in self.files:
            with np.load(f) as z:
                out.append(int(z["is_crash"]))
        return out

    def __getitem__(self, idx: int):
        with np.load(self.files[idx]) as z:
            emb = z["emb"].astype(np.float32)
            y_pre = z["y_pre"].astype(np.float32)
            y_post = z["y_post"].astype(np.float32)
            is_crash = int(z["is_crash"])
            times = z["times"].astype(np.float32)
            t_gt = float(z["t_gt"])
        t = emb.shape[0]
        if t > self.max_len:
            # keep neighborhood around GT for crash; else random crop
            if is_crash and t_gt >= 0:
                center = int(np.argmin(np.abs(times - t_gt)))
                start = max(0, min(center - self.max_len // 2, t - self.max_len))
            else:
                start = random.randint(0, t - self.max_len)
            sl = slice(start, start + self.max_len)
            emb, y_pre, y_post, times = emb[sl], y_pre[sl], y_post[sl], times[sl]
            t = self.max_len
        return {
            "emb": torch.from_numpy(emb),
            "y_pre": torch.from_numpy(y_pre),
            "y_post": torch.from_numpy(y_post),
            "is_crash": torch.tensor(is_crash, dtype=torch.float32),
            "length": t,
            "video_id": self.files[idx].stem,
        }


def collate(batch):
    max_t = max(b["length"] for b in batch)
    d = batch[0]["emb"].shape[-1]
    B = len(batch)
    emb = torch.zeros(B, max_t, d)
    y_pre = torch.zeros(B, max_t)
    y_post = torch.zeros(B, max_t)
    mask = torch.zeros(B, max_t, dtype=torch.bool)
    crash = torch.zeros(B)
    ids = []
    for i, b in enumerate(batch):
        t = b["length"]
        emb[i, :t] = b["emb"]
        y_pre[i, :t] = b["y_pre"]
        y_post[i, :t] = b["y_post"]
        mask[i, :t] = True
        crash[i] = b["is_crash"]
        ids.append(b["video_id"])
    return {
        "emb": emb,
        "y_pre": y_pre,
        "y_post": y_post,
        "valid_mask": mask,
        "is_crash": crash,
        "video_id": ids,
    }


def masked_bce(pred, target, mask):
    pred = pred[mask]
    target = target[mask]
    if pred.numel() == 0:
        return pred.new_zeros(())
    return F.binary_cross_entropy(pred, target)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    crash_logits, crash_gt = [], []
    pre_losses, post_losses = [], []
    for batch in loader:
        emb = batch["emb"].to(device)
        mask = batch["valid_mask"].to(device)
        out = model(emb, mask)
        pre_losses.append(float(masked_bce(out["q_pre"], batch["y_pre"].to(device), mask)))
        post_losses.append(float(masked_bce(out["q_post"], batch["y_post"].to(device), mask)))
        crash_logits.append(torch.sigmoid(out["crash_logit"]).cpu())
        crash_gt.append(batch["is_crash"])
    probs = torch.cat(crash_logits).numpy()
    gt = torch.cat(crash_gt).numpy()
    pred = (probs >= 0.5).astype(np.float32)
    crash_mask = gt >= 0.5
    non_mask = ~crash_mask
    rec = float(pred[crash_mask].mean()) if crash_mask.any() else 0.0
    spec = float((1 - pred[non_mask]).mean()) if non_mask.any() else 0.0
    return {
        "pre_bce": float(np.mean(pre_losses)) if pre_losses else 0.0,
        "post_bce": float(np.mean(post_losses)) if post_losses else 0.0,
        "recall": rec,
        "specificity": spec,
        "bal_acc": 0.5 * (rec + spec),
        "n_crash": int(crash_mask.sum()),
        "n_non": int(non_mask.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default=str(HERE / "cache_emb"))
    p.add_argument("--output_dir", default=str(HERE / "checkpoints/cumcrash_v1"))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda_crash", type=float, default=1.0)
    p.add_argument("--lambda_pre", type=float, default=1.0)
    p.add_argument("--lambda_post", type=float, default=1.0)
    p.add_argument("--gru_hidden", type=int, default=256)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SeqEmbDataset(Path(args.cache_dir) / "train", max_len=args.max_len)
    val_ds = SeqEmbDataset(Path(args.cache_dir) / "val", max_len=args.max_len)
    labels = train_ds.labels()
    n_pos = max(sum(labels), 1)
    n_neg = max(len(labels) - n_pos, 1)
    weights = [1.0 / n_pos if y == 1 else 1.0 / n_neg for y in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
    )

    # peek emb dim
    d = train_ds[0]["emb"].shape[-1]
    model = CumCrashModel(emb_dim=d, gru_hidden=args.gru_hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    best = -1.0
    history = []
    print(f"Device={device} train={len(train_ds)} val={len(val_ds)} emb_dim={d}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            emb = batch["emb"].to(device)
            mask = batch["valid_mask"].to(device)
            y_pre = batch["y_pre"].to(device)
            y_post = batch["y_post"].to(device)
            y_c = batch["is_crash"].to(device)
            out = model(emb, mask)
            loss_pre = masked_bce(out["q_pre"], y_pre, mask)
            loss_post = masked_bce(out["q_post"], y_post, mask)
            loss_crash = F.binary_cross_entropy_with_logits(out["crash_logit"], y_c)
            loss = (
                args.lambda_pre * loss_pre
                + args.lambda_post * loss_post
                + args.lambda_crash * loss_crash
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

        metrics = evaluate(model, val_loader, device)
        line = (
            f"[Epoch {epoch}] loss={np.mean(losses):.4f} "
            f"bal_acc={metrics['bal_acc']:.3f} rec={metrics['recall']:.3f} "
            f"spec={metrics['specificity']:.3f} "
            f"pre_bce={metrics['pre_bce']:.4f} post_bce={metrics['post_bce']:.4f}"
        )
        print(line)
        with (out_dir / "train.log").open("a") as f:
            f.write(line + "\n")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        if metrics["bal_acc"] >= best:
            best = metrics["bal_acc"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "args": vars(args)}, out_dir / "best.pt")
            print(f"  saved best.pt bal_acc={best:.3f}")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. best_bal_acc={best:.3f} -> {out_dir}")


if __name__ == "__main__":
    main()
