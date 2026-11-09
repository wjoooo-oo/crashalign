#!/usr/bin/env python3
# 0727_eval_accident_t_complete.py
"""Complete ACCIDENT T comparison: current VideoMAE T vs phase+video T head.

在 ACCIDENT matched YouTube-split crash 上完整对比 onset 定位：
  A0) VideoMAE-clean 分类头（零样本，现有 T 配方 / 同 encoder emb）
  A1) VideoMAE-head 在 ACCIDENT train 上微调（公平「当前 T 方法」）
  B0) emb_only MLP T head（仅 video）
  B1) phase_cond MLP T head（video ‖ stopgrad pred phase）
  B2) phase_oracle（video ‖ GT phase，上限）

数据：cache_emb_accident（与 extract 使用 checkpoints_temporal_clean）。
指标：MAE / median / ≤1s / ≤2s / T-score(σ=2)。
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import VideoMAEForVideoClassification

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from crashalign_paths import accident_root  # noqa: E402
from t_stage import (  # noqa: E402
    SIGMA_T,
    CrashEmbDataset,
    OnsetScoreHead,
    build_t_input,
    collate,
    decode_onset,
    evaluate_t_head,
    load_stage1_phase_model,
    masked_mse,
    predict_phase_prob,
    t_score,
)


class VideoMAEOnsetHead(nn.Module):
    """Linear risk head matching VideoMAEForVideoClassification.classifier."""

    def __init__(self, in_dim: int = 768):
        super().__init__()
        self.classifier = nn.Linear(in_dim, 1)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.classifier(emb).squeeze(-1)

    @classmethod
    def from_videomae_ckpt(cls, ckpt_dir: Path, device: torch.device) -> "VideoMAEOnsetHead":
        base = VideoMAEForVideoClassification.from_pretrained(str(ckpt_dir))
        head = cls(in_dim=base.config.hidden_size)
        head.classifier.load_state_dict(base.classifier.state_dict())
        return head.to(device)


@torch.no_grad()
def evaluate_emb_head(
    head: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    use_sigmoid: bool = True,
) -> dict:
    """Eval a per-window score head on emb only (VideoMAE-style)."""
    head.eval()
    errs, ts = [], []
    for batch in loader:
        emb = batch["emb"].to(device)
        mask = batch["valid_mask"].to(device)
        times = batch["times"].to(device)
        t_gt = batch["t_gt"].to(device)
        logits = head(emb)
        scores = torch.sigmoid(logits) if use_sigmoid else logits
        t_pred = decode_onset(scores, times, mask)
        err = (t_pred - t_gt).abs()
        errs.extend(err.cpu().tolist())
        ts.extend([t_score(float(e)) for e in err.cpu().tolist()])
    errs = np.asarray(errs, dtype=np.float64)
    return {
        "n": int(len(errs)),
        "mae": float(errs.mean()),
        "median_ae": float(np.median(errs)),
        "within_1s": float((errs <= 1.0).mean()),
        "within_2s": float((errs <= 2.0).mean()),
        "t_score": float(np.mean(ts)),
    }


def train_head(
    mode: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    phase_model,
    device: torch.device,
    args: argparse.Namespace,
    out_dir: Path,
    *,
    init_videomae: Path | None = None,
) -> dict:
    """Train onset head; mode in {videomae_ft, emb_only, phase_cond, phase_oracle}."""
    if mode == "videomae_ft":
        if init_videomae is not None:
            head: nn.Module = VideoMAEOnsetHead.from_videomae_ckpt(init_videomae, device)
        else:
            head = VideoMAEOnsetHead().to(device)
        in_dim = 768
    else:
        in_dim = 768 if mode == "emb_only" else 772
        head = OnsetScoreHead(in_dim, hidden=args.hidden, dropout=args.dropout).to(device)

    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_mae, best_state, bad = 1e9, None, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{mode} ep{epoch}", leave=False):
            emb = batch["emb"].to(device)
            mask = batch["valid_mask"].to(device)
            y = batch["y_onset"].to(device)
            gt_phase = batch["gt_phase"].to(device)
            if mode == "videomae_ft":
                logits = head(emb)
            else:
                with torch.no_grad():
                    phase_feat = predict_phase_prob(phase_model, emb, mask, mode, gt_phase)
                x = build_t_input(emb, phase_feat)
                logits = head(x)
            pred = torch.sigmoid(logits)
            loss = masked_mse(pred, y, mask)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))

        if mode == "videomae_ft":
            val = evaluate_emb_head(head, val_loader, device)
        else:
            val = evaluate_t_head(head, phase_model, val_loader, device, mode)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val})
        print(
            f"[{mode} {epoch}] loss={np.mean(losses):.4f} "
            f"val_mae={val['mae']:.3f} @1s={val['within_1s']:.3f} T={val['t_score']:.3f}"
        )
        if val["mae"] < best_mae - 1e-4:
            best_mae = val["mae"]
            bad = 0
            best_state = {
                "head": head.state_dict(),
                "mode": mode,
                "in_dim": in_dim,
                "val": val,
                "epoch": epoch,
            }
            torch.save(best_state, out_dir / f"{mode}_best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{mode}] early stop at epoch {epoch}")
                break

    assert best_state is not None
    head.load_state_dict(best_state["head"])
    if mode == "videomae_ft":
        test = evaluate_emb_head(head, args._test_loader, device)
    else:
        test = evaluate_t_head(head, phase_model, args._test_loader, device, mode)
    return {
        "mode": mode,
        "best_epoch": best_state["epoch"],
        "val": best_state["val"],
        "test": test,
        "history": history,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", default=str(HERE / "cache_emb_accident"))
    p.add_argument(
        "--videomae_ckpt",
        default=str(accident_root() / "vit/checkpoints_temporal_clean/best_model"),
        help="Current T backbone+head (also used to extract embs)",
    )
    p.add_argument(
        "--phase_ckpt",
        default=str(HERE / "checkpoints/accident_matched/phase_only/best.pt"),
    )
    p.add_argument("--output_dir", default=str(HERE / "results/accident_t_complete"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    cache_root = Path(args.cache_root)
    train_ds = CrashEmbDataset(cache_root / "train")
    val_ds = CrashEmbDataset(cache_root / "val")
    test_ds = CrashEmbDataset(cache_root / "test")
    print(
        f"Device={device} ACCIDENT matched crash: "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate,
    )
    args._test_loader = test_loader

    report: dict = {
        "protocol": "ACCIDENT matched YouTube-split crash-only onset T",
        "cache": str(cache_root),
        "videomae_ckpt": args.videomae_ckpt,
        "phase_ckpt": args.phase_ckpt,
        "sigma_t": SIGMA_T,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "runs": [],
    }

    # ----- A0: zero-shot VideoMAE-clean head -----
    print("\n===== A0 videomae_clean_zeroshot =====")
    vm_head = VideoMAEOnsetHead.from_videomae_ckpt(Path(args.videomae_ckpt), device)
    a0_val = evaluate_emb_head(vm_head, val_loader, device)
    a0_test = evaluate_emb_head(vm_head, test_loader, device)
    report["runs"].append(
        {"mode": "videomae_clean_zeroshot", "best_epoch": None, "val": a0_val, "test": a0_test}
    )
    print(f"TEST A0 MAE={a0_test['mae']:.3f} T={a0_test['t_score']:.3f}")

    # ----- A1: finetune VideoMAE head on ACCIDENT -----
    print("\n===== A1 videomae_ft (ACCIDENT) =====")
    a1 = train_head(
        "videomae_ft",
        train_loader,
        val_loader,
        None,
        device,
        args,
        out_dir,
        init_videomae=Path(args.videomae_ckpt),
    )
    report["runs"].append(a1)
    print(f"TEST A1 MAE={a1['test']['mae']:.3f} T={a1['test']['t_score']:.3f}")

    # ----- B: emb_only / phase_cond / phase_oracle -----
    phase_model = load_stage1_phase_model(Path(args.phase_ckpt), device)
    print(f"Frozen Stage-I phase: {args.phase_ckpt}")
    for mode in ("emb_only", "phase_cond", "phase_oracle"):
        print(f"\n===== B {mode} =====")
        run = train_head(mode, train_loader, val_loader, phase_model, device, args, out_dir)
        report["runs"].append(run)
        te = run["test"]
        print(f"TEST {mode} MAE={te['mae']:.3f} T={te['t_score']:.3f}")

    (out_dir / "0727_accident_t_complete_metrics.json").write_text(json.dumps(report, indent=2))

    # Markdown / 结果表
    order = [
        "videomae_clean_zeroshot",
        "videomae_ft",
        "emb_only",
        "phase_cond",
        "phase_oracle",
    ]
    name_map = {
        "videomae_clean_zeroshot": "A0 VideoMAE-clean (zero-shot)",
        "videomae_ft": "A1 VideoMAE-head FT on ACCIDENT",
        "emb_only": "B0 emb-only T head",
        "phase_cond": "B1 phase+video T head (pred phase)",
        "phase_oracle": "B2 phase+video T head (GT phase oracle)",
    }
    by = {r["mode"]: r for r in report["runs"]}
    lines = [
        "# Complete ACCIDENT T comparison (matched YouTube-split crash)",
        "",
        f"- Train/val/test: **{len(train_ds)}/{len(val_ds)}/{len(test_ds)}** crash videos",
        f"- Encoder: `{args.videomae_ckpt}` (same as emb cache)",
        f"- Stage-I phase (frozen): `{args.phase_ckpt}`",
        f"- Decode: risk-peak over 2.0s/0.25s windows; T-score σ={SIGMA_T}",
        "",
        "| Method | Val MAE | Test MAE | Med AE | ≤1s | ≤2s | T-score |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in order:
        r = by[m]
        te, va = r["test"], r["val"]
        lines.append(
            f"| {name_map[m]} | {va['mae']:.3f} | {te['mae']:.3f} | {te['median_ae']:.3f} | "
            f"{te['within_1s']:.3f} | {te['within_2s']:.3f} | {te['t_score']:.3f} |"
        )

    a1t = by["videomae_ft"]["test"]
    b0 = by["emb_only"]["test"]
    b1 = by["phase_cond"]["test"]
    b2 = by["phase_oracle"]["test"]
    lines += [
        "",
        "## Takeaway",
        "",
        f"- **Current T adapted to ACCIDENT (A1)** test MAE={a1t['mae']:.3f}, T={a1t['t_score']:.3f}.",
        f"- **phase+video pred (B1)** vs A1: ΔMAE={b1['mae']-a1t['mae']:+.3f}s, "
        f"ΔT={b1['t_score']-a1t['t_score']:+.3f}.",
        f"- **phase+video pred (B1)** vs emb-only (B0): ΔMAE={b1['mae']-b0['mae']:+.3f}s.",
        f"- **GT-phase oracle (B2)** MAE={b2['mae']:.3f} — conditioning channel works; "
        "pred-phase quality is the bottleneck.",
        "",
    ]
    md = out_dir / "0727_accident_t_complete.md"
    md.write_text("\n".join(lines))
    print(f"\nWrote {md}")


if __name__ == "__main__":
    main()
