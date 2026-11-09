#!/usr/bin/env python3
"""Train Task-1 cumulative phase crash detector on cached embeddings."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labels import phase_ids_from_cum  # noqa: E402
from losses import balanced_bce_with_logits, balanced_ce_with_logits  # noqa: E402
from model import Task1CumPhaseModel  # noqa: E402
from sampling import crop_or_pad  # noqa: E402


class SeqEmbDataset(Dataset):
    def __init__(self, cache_dir: Path, fixed_len: int = 76, train: bool = True):
        self.files = sorted(cache_dir.glob("*.npz"))
        self.fixed_len = fixed_len
        self.train = train
        self.rng = np.random.default_rng(42)
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
        with np.load(self.files[idx], allow_pickle=True) as z:
            emb = z["emb"].astype(np.float32)
            y_a = z["y_a"].astype(np.float32)
            y_b = z["y_b"].astype(np.float32)
            phase_valid = z["phase_valid"].astype(bool)
            is_crash = int(z["is_crash"])
            times = z["times"].astype(np.float32)
        mode = "random" if self.train else "center"
        emb, y_a, y_b, phase_valid, times, valid = crop_or_pad(
            emb, y_a, y_b, phase_valid, times, self.fixed_len, mode=mode
        )
        # 4-class targets keep impact supervised; padding stays unlabeled (-1)
        y_phase = phase_ids_from_cum(y_a, y_b, phase_valid, valid)
        # cumulative a/b losses only on non-impact AND non-pad
        phase_valid = phase_valid & valid
        return {
            "emb": torch.from_numpy(emb),
            "y_a": torch.from_numpy(y_a),
            "y_b": torch.from_numpy(y_b),
            "y_phase": torch.from_numpy(y_phase),
            "phase_valid": torch.from_numpy(phase_valid),
            "valid_mask": torch.from_numpy(valid),
            "is_crash": torch.tensor(is_crash, dtype=torch.float32),
            "length": self.fixed_len,
            "video_id": self.files[idx].stem,
        }


def collate(batch):
    emb = torch.stack([b["emb"] for b in batch], dim=0)
    return {
        "emb": emb,
        "y_a": torch.stack([b["y_a"] for b in batch], dim=0),
        "y_b": torch.stack([b["y_b"] for b in batch], dim=0),
        "y_phase": torch.stack([b["y_phase"] for b in batch], dim=0),
        "phase_valid": torch.stack([b["phase_valid"] for b in batch], dim=0),
        "valid_mask": torch.stack([b["valid_mask"] for b in batch], dim=0),
        "is_crash": torch.stack([b["is_crash"] for b in batch], dim=0),
    }


def phase_loss(model, out, batch, device, valid_mask):
    """Auxiliary phase loss for either phase head parameterization."""
    if model.phase_mode == "softmax4":
        # padding excluded; impact is a supervised class
        y_phase = batch["y_phase"].to(device)
        return balanced_ce_with_logits(out["phase_logits"], y_phase.clamp(min=0), valid_mask)
    phase_mask = batch["phase_valid"].to(device) & valid_mask
    l_a = balanced_bce_with_logits(out["a_logits"], batch["y_a"].to(device), phase_mask)
    l_b = balanced_bce_with_logits(out["b_logits"], batch["y_b"].to(device), phase_mask)
    return l_a + l_b


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    probs, gts = [], []
    la_list = []
    for batch in loader:
        emb = batch["emb"].to(device)
        mask = batch["valid_mask"].to(device)
        out = model(emb, mask)
        la_list.append(float(phase_loss(model, out, batch, device, mask)))
        probs.append(torch.sigmoid(out["crash_logit"]).cpu())
        gts.append(batch["is_crash"])
    probs = torch.cat(probs).numpy()
    gts = torch.cat(gts).numpy()
    # threshold sweep for bal_acc
    best_thr, best_bal = 0.5, -1.0
    for thr in np.linspace(0.05, 0.95, 37):
        pred = (probs >= thr).astype(np.float32)
        crash = gts >= 0.5
        non = ~crash
        rec = pred[crash].mean() if crash.any() else 0.0
        spec = (1 - pred[non]).mean() if non.any() else 0.0
        bal = 0.5 * (rec + spec)
        if bal > best_bal:
            best_bal, best_thr = float(bal), float(thr)
    pred = (probs >= best_thr).astype(np.float32)
    crash = gts >= 0.5
    non = ~crash
    rec = float(pred[crash].mean()) if crash.any() else 0.0
    spec = float((1 - pred[non]).mean()) if non.any() else 0.0
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score

        auroc = float(roc_auc_score(gts, probs)) if len(np.unique(gts)) > 1 else None
        ap = float(average_precision_score(gts, probs)) if len(np.unique(gts)) > 1 else None
    except Exception:
        auroc = ap = None
    return {
        "bal_acc": 0.5 * (rec + spec),
        "recall": rec,
        "specificity": spec,
        "best_thr": best_thr,
        "auroc": auroc,
        "ap": ap,
        "phase_loss": float(np.mean(la_list)) if la_list else 0.0,
        "n_crash": int(crash.sum()),
        "n_non": int(non.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", default=str(HERE / "cache_emb"))
    p.add_argument("--output_dir", default=str(HERE / "checkpoints/task1_d"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--lambda_phase", type=float, default=0.2)
    p.add_argument("--gru_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument(
        "--crash_feat",
        default="full",
        choices=(
            "full",
            "phase_only",
            "h_only",
            "phase_unordered",
            "pattern_gated",
            "fp_gated",
        ),
        help="full|phase_only|h_only|phase_unordered|pattern_gated|fp_gated",
    )
    p.add_argument(
        "--phase_mode",
        default="cum",
        choices=("cum", "softmax4"),
        help="cum=legacy a_t/b_t sigmoids; softmax4=4-way phase logits + masked CE",
    )
    p.add_argument("--fixed_len", type=int, default=76, help="Fixed #windows; blocks length cue")
    p.add_argument(
        "--dataset",
        default="legacy",
        choices=("legacy", "accident_matched"),
        help="legacy=pad/crop cache; accident_matched=same-video pos/neg fixed-L clips",
    )
    p.add_argument(
        "--matched_data_dir",
        default=str(HERE / "data_accident_matched"),
        help="Directory with {split}_matched.jsonl for --dataset accident_matched",
    )
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "accident_matched":
        # ACCIDENT-only matched protocol: exactly L windows, no padding.
        # 仅 ACCIDENT 的 matched 协议：定长 L、无 padding。
        import importlib.util

        _spec = importlib.util.spec_from_file_location(
            "matched_dataset", HERE / "0726_matched_dataset.py"
        )
        _mod = importlib.util.module_from_spec(_spec)
        assert _spec.loader is not None
        _spec.loader.exec_module(_mod)
        MatchedAccidentDataset = _mod.MatchedAccidentDataset
        mdir = Path(args.matched_data_dir)
        train_ds = MatchedAccidentDataset(
            mdir / "train_matched.jsonl",
            Path(args.cache_dir) / "train",
            fixed_len=args.fixed_len,
            train=True,
            seed=args.seed,
        )
        val_ds = MatchedAccidentDataset(
            mdir / "val_matched.jsonl",
            Path(args.cache_dir) / "val",
            fixed_len=args.fixed_len,
            train=False,
            seed=args.seed,
        )
    else:
        train_ds = SeqEmbDataset(Path(args.cache_dir) / "train", fixed_len=args.fixed_len, train=True)
        val_ds = SeqEmbDataset(Path(args.cache_dir) / "val", fixed_len=args.fixed_len, train=False)
    labels = train_ds.labels()
    n_pos = max(sum(labels), 1)
    n_neg = max(len(labels) - n_pos, 1)
    weights = [0.5 / n_pos if y == 1 else 0.5 / n_neg for y in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )

    d = train_ds[0]["emb"].shape[-1]
    model = Task1CumPhaseModel(
        emb_dim=d,
        gru_hidden=args.gru_hidden,
        dropout=args.dropout,
        crash_feat=args.crash_feat,
        phase_mode=args.phase_mode,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=device.type == "cuda")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(
        f"Device={device} train={len(train_ds)} val={len(val_ds)} "
        f"crash={n_pos} non={n_neg} emb_dim={d} "
        f"crash_feat={args.crash_feat} phase_mode={args.phase_mode} "
        f"lambda_phase={args.lambda_phase}"
    )

    best = -1.0
    bad = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        n_crash_b = n_non_b = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            emb = batch["emb"].to(device)
            mask = batch["valid_mask"].to(device)
            y_c = batch["is_crash"].to(device)
            n_crash_b += int((y_c > 0.5).sum())
            n_non_b += int((y_c <= 0.5).sum())
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                out = model(emb, mask)
                l_det = F.binary_cross_entropy_with_logits(out["crash_logit"], y_c)
                if args.lambda_phase > 0:
                    l_ph = phase_loss(model, out, batch, device, mask)
                else:
                    l_ph = torch.zeros((), device=device)
                loss = l_det + args.lambda_phase * l_ph
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")
        sched.step()

        metrics = evaluate(model, val_loader, device)
        line = (
            f"[Epoch {epoch}] loss={np.mean(losses):.4f} "
            f"bal_acc={metrics['bal_acc']:.3f} thr={metrics['best_thr']:.2f} "
            f"rec={metrics['recall']:.3f} spec={metrics['specificity']:.3f} "
            f"auroc={metrics['auroc']} "
            f"batch_crash/non={n_crash_b}/{n_non_b}"
        )
        print(line)
        with (out_dir / "train.log").open("a") as f:
            f.write(line + "\n")
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        if metrics["bal_acc"] >= best:
            best = metrics["bal_acc"]
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "best_thr": metrics["best_thr"],
                    "args": vars(args),
                    "emb_dim": d,
                },
                out_dir / "best.pt",
            )
            (out_dir / "best_threshold.json").write_text(
                json.dumps({"best_thr": metrics["best_thr"], "val_bal_acc": best}, indent=2)
            )
            print(f"  saved best.pt bal_acc={best:.3f} thr={metrics['best_thr']:.2f}")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stop at epoch {epoch}")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Done. best_bal_acc={best:.3f} -> {out_dir}")


if __name__ == "__main__":
    main()
