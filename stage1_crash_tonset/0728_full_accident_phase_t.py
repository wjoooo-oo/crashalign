#!/usr/bin/env python3
# 0728_full_accident_phase_t.py
"""Onset localization on the COMPLETE ACCIDENT dataset: T head vs phase vs both.

Protocol / 协议:
  All annotated crash videos, 1419/201/407 train/val/test (source-video split,
  seed 42 — the same split used by the matched detection protocol, just without
  the clean-normal-negative filter that dropped 1266 videos irrelevant to T).
  全部带标注的事故视频；split 与 matched 协议一致，
  只是去掉了那条与 T 无关、剔除了 1266 个视频的 clean-normal 负样本约束。

Methods compared / 对比方法:
  t_only        emb -> T head, Gaussian soft-onset regression (the original T)
                原始方案：仅 T 头，高斯软 onset 回归
  phase_only    emb -> 4-phase head; onset read off the P(impact) peak.
                No T head at all.
                仅 phase 头，用 P(impact) 峰值读出 onset，完全没有 T 头
  phase_cat_T   T head takes [emb || stopgrad predicted phase] (the B1 coupling)
                T 头输入 [emb ‖ 冻结的预测 phase]（B1 的接法）
  phase_prior_T decode-time fusion, no extra training:
                score_t = T_t^(1-w) * blur(P(impact))_t^w, w chosen on val.
                解码期融合，无需额外训练；权重 w 在 val 上选。

The prior coupling exists because E1 (0728_phase_t_diagnostics.py) showed a
blurred phase costs nothing while 4 extra dims concatenated onto 768 get
ignored once the phase is imperfect.
之所以加先验接法：E1 表明 phase 模糊无害，
但 4 维拼进 768 维后，一旦 phase 不完美就会被忽略。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t_stage import (  # noqa: E402
    CrashEmbDataset,
    OnsetScoreHead,
    collate,
    decode_onset,
    masked_mse,
    onset_metrics,
    phase_ids_from_times,
    t_score,
)

# The phase-progression head lives in the E2 training script, whose `0728_` name
# cannot be imported normally, so it is loaded by path.
# 阶段进程头位于 E2 训练脚本中，其 0728_ 文件名无法用普通 import 导入，故按路径加载。
_spec = importlib.util.spec_from_file_location("phase_prog", HERE / "0728_train_phase_progression.py")
_prog = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_prog)

PhaseProgressionModel = _prog.PhaseProgressionModel
evaluate_phase = _prog.evaluate_phase
phase_loss = _prog.phase_loss


# --------------------------------------------------------------------------
# stage A: phase head (the v1 recipe: crash-only, full-length, 4-way softmax)
# --------------------------------------------------------------------------
def train_phase_head(loaders, device, args) -> tuple[nn.Module, dict]:
    train_loader, val_loader, test_loader = loaders
    out_dir = Path(args.ckpt_root) / "phase_head"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_dim = next(iter(val_loader))["emb"].shape[-1]
    model = PhaseProgressionModel(
        emb_dim=emb_dim, gru_hidden=args.gru_hidden, dropout=args.dropout, head="softmax4"
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.phase_lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def prob_fn(emb, mask):
        model.eval()
        return model(emb, mask)["phase_prob"]

    best, bad = 1e9, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"phase ep{epoch}", leave=False):
            emb = batch["emb"].to(device)
            mask = batch["valid_mask"].to(device)
            y_ids = phase_ids_from_times(batch["times"].to(device), batch["t_gt"].to(device))
            loss = phase_loss(model, model(emb, mask), y_ids, mask)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        sched.step()
        val = evaluate_phase(prob_fn, val_loader, device)
        sel = val["best_onset_mae"]
        print(
            f"[phase {epoch}] loss={np.mean(losses):.4f} val_onsetMAE={sel:.3f} "
            f"impactF1={val['best_impact_f1']:.3f}"
        )
        if sel < best - 1e-4:
            best, bad = sel, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val},
                       out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[phase] early stop at epoch {epoch}")
                break
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    test = evaluate_phase(prob_fn, test_loader, device)
    print(f"TEST [phase_only] onsetMAE={test['best_onset_mae']:.3f}")
    return model, test


@torch.no_grad()
def phase_prob_cache(model, loader, device) -> dict[str, np.ndarray]:
    """Precompute P(impact) per video so the T stage never re-runs the phase net.

    预先缓存每个视频的 P(impact)，避免 T 阶段重复推理 phase 网络。
    """
    out = {}
    for batch in loader:
        prob = model(batch["emb"].to(device), batch["valid_mask"].to(device))
        p_imp = prob["phase_prob"][..., 2].cpu().numpy()
        n = batch["valid_mask"].sum(dim=1).numpy()
        for i, vid in enumerate(batch["video_id"]):
            out[vid] = p_imp[i, : int(n[i])]
    return out


# --------------------------------------------------------------------------
# stage B: T heads
# --------------------------------------------------------------------------
@torch.no_grad()
def t_head_scores(head, phase_model, loader, device, use_phase: bool):
    """Per-video sigmoid scores from the T head, plus times and ground truth."""
    head.eval()
    recs = []
    for batch in loader:
        emb = batch["emb"].to(device)
        mask = batch["valid_mask"].to(device)
        x = emb
        if use_phase:
            feat = phase_model(emb, mask)["phase_prob"].detach()
            x = torch.cat([emb, feat], dim=-1)
        s = torch.sigmoid(head(x)).cpu().numpy()
        n = mask.sum(dim=1).cpu().numpy()
        times = batch["times"].numpy()
        t_gt = batch["t_gt"].numpy()
        for i, vid in enumerate(batch["video_id"]):
            L = int(n[i])
            recs.append(
                {"video_id": vid, "score": s[i, :L], "times": times[i, :L], "t_gt": float(t_gt[i])}
            )
    return recs


def train_t_head(loaders, device, args, phase_model=None, use_phase: bool = False,
                 tag: str = "t_only") -> tuple[nn.Module, dict]:
    train_loader, val_loader, test_loader = loaders
    in_dim = 768 + (4 if use_phase else 0)
    head = OnsetScoreHead(in_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.t_lr, weight_decay=1e-4)
    out_dir = Path(args.ckpt_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    def eval_split(loader):
        recs = t_head_scores(head, phase_model, loader, device, use_phase)
        errs = [
            abs(float(r["times"][int(np.argmax(r["score"]))]) - r["t_gt"]) for r in recs
        ]
        return onset_metrics(errs)

    best, bad = 1e9, 0
    for epoch in range(1, args.epochs + 1):
        head.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{tag} ep{epoch}", leave=False):
            emb = batch["emb"].to(device)
            mask = batch["valid_mask"].to(device)
            y = batch["y_onset"].to(device)
            x = emb
            if use_phase:
                with torch.no_grad():
                    feat = phase_model(emb, mask)["phase_prob"].detach()
                x = torch.cat([emb, feat], dim=-1)
            loss = masked_mse(torch.sigmoid(head(x)), y, mask)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        val = eval_split(val_loader)
        print(
            f"[{tag} {epoch}] loss={np.mean(losses):.4f} "
            f"val_MAE={val['mae']:.3f} T={val['t_score']:.3f}"
        )
        if val["mae"] < best - 1e-4:
            best, bad = val["mae"], 0
            torch.save({"head": head.state_dict(), "epoch": epoch, "val": val},
                       out_dir / "best.pt")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{tag}] early stop at epoch {epoch}")
                break
    head.load_state_dict(torch.load(out_dir / "best.pt", map_location=device)["head"])
    test = eval_split(test_loader)
    print(f"TEST [{tag}] MAE={test['mae']:.3f} T={test['t_score']:.3f}")
    return head, test


# --------------------------------------------------------------------------
# stage C: decode-time fusion of the T score with a phase prior
# --------------------------------------------------------------------------
def smooth_1d(x: np.ndarray, sigma_win: float) -> np.ndarray:
    """Gaussian smoothing along time with edge padding.

    沿时间做高斯平滑（边缘复制填充）。E1 表明把 phase 展宽不会损失定位精度，
    反而让峰更平滑、更易与 T 分数相乘。
    """
    if sigma_win <= 0 or len(x) < 2:
        return x
    radius = max(1, int(round(3 * sigma_win)))
    k = np.exp(-(np.arange(-radius, radius + 1) ** 2) / (2 * sigma_win ** 2))
    k /= k.sum()
    padded = np.pad(x, radius, mode="edge")
    return np.convolve(padded, k, mode="valid")


def fuse_and_score(recs, phase_cache: dict[str, np.ndarray], w: float,
                   blur_win: float) -> dict:
    """score_t = T_t^(1-w) * blur(P(impact))_t^w, then peak-pick.

    对数域加权几何平均后取峰。w=0 等价纯 T，w=1 等价纯 phase。
    """
    eps = 1e-6
    errs = []
    for r in recs:
        prior = smooth_1d(phase_cache[r["video_id"]], blur_win)
        fused = (1.0 - w) * np.log(r["score"] + eps) + w * np.log(prior + eps)
        errs.append(abs(float(r["times"][int(np.argmax(fused))]) - r["t_gt"]))
    return onset_metrics(errs)


def sweep_fusion(val_recs, test_recs, phase_cache_val, phase_cache_test,
                 weights, blur_win) -> dict:
    """Pick the fusion weight on val, then report the test result at that weight.

    在 val 上选融合权重，再用该权重报告 test，避免在测试集上调参。
    """
    table = []
    for w in weights:
        v = fuse_and_score(val_recs, phase_cache_val, w, blur_win)
        table.append({"w": w, "val": v})
    best = min(table, key=lambda r: r["val"]["mae"])
    test = fuse_and_score(test_recs, phase_cache_test, best["w"], blur_win)
    return {"sweep": table, "best_w": best["w"], "val": best["val"], "test": test}


# --------------------------------------------------------------------------
def coverage_report(loader) -> dict:
    """How often the annotated onset even falls inside the window grid.

    标注 onset 落在窗口网格之外的比例——它给出 MAE 的不可约下界。
    """
    inside, offsets, n = 0, [], 0
    for batch in loader:
        L = batch["valid_mask"].sum(dim=1).numpy()
        times, t_gt = batch["times"].numpy(), batch["t_gt"].numpy()
        for i in range(len(t_gt)):
            lo, hi = float(times[i, 0]), float(times[i, int(L[i]) - 1])
            n += 1
            if lo <= t_gt[i] <= hi:
                inside += 1
                offsets.append(0.0)
            else:
                offsets.append(min(abs(t_gt[i] - lo), abs(t_gt[i] - hi)))
    return {
        "n": n,
        "onset_inside_grid": inside / max(n, 1),
        "mean_unreachable_error": float(np.mean(offsets)),
    }


def write_report(res: dict, out_path: Path) -> None:
    cov = res["coverage"]
    rows = [
        ("t_only (original T head)", res["t_only"]),
        ("phase_only (no T head)", res["phase_only_onset"]),
        ("phase_cat_T", res["phase_cat_T"]),
        (f"phase_prior_T (w={res['phase_prior_T']['best_w']:.2f})",
         res["phase_prior_T"]["test"]),
    ]
    lines = [
        "# Complete ACCIDENT — onset localization",
        "",
        f"Videos: train {res['n_train']} / val {res['n_val']} / test {res['n_test']} "
        "(all annotated crash videos, source-video split).",
        "Frozen encoder `vit/checkpoints_temporal_clean/best_model`; "
        "2.0 s windows, 0.25 s stride; T-score σ = 2 s.",
        "",
        "| Method | Test MAE | Median AE | ≤1s | ≤2s | T-score |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in rows:
        lines.append(
            f"| {name} | {m['mae']:.3f} s | {m['median_ae']:.3f} s | "
            f"{m['within_1s']:.3f} | {m['within_2s']:.3f} | **{m['t_score']:.3f}** |"
        )

    lines += [
        "",
        "## Fusion weight sweep (chosen on val, applied to test)",
        "",
        "`score_t = T_t^(1-w) · blur(P(impact))_t^w`; w=0 is the T head alone, "
        "w=1 is the phase head alone.",
        "",
        "| w | Val MAE | Val T-score |",
        "|---:|---:|---:|",
    ]
    for r in res["phase_prior_T"]["sweep"]:
        lines.append(
            f"| {r['w']:.2f} | {r['val']['mae']:.3f} s | {r['val']['t_score']:.3f} |"
        )

    lines += [
        "",
        "## Phase-head quality on the full test set",
        "",
        f"- onset MAE from P(impact) peak: {res['phase_only']['onset_from_impact_peak']['mae']:.3f} s",
        f"- impact F1 (best decoder): {res['phase_only']['best_impact_f1']:.3f}",
        "- phase switches per video (cum50 decode, median): "
        f"{res['phase_only']['decode']['cum50']['temporal_consistency']['switches_per_video']['median']:.0f}"
        f" vs ground truth {res['phase_only']['gt_switches_per_video']['median']:.0f}",
        "- legal transition rate: "
        f"{res['phase_only']['decode']['cum50']['temporal_consistency']['legal_transition_rate']:.3f}",
        "",
        "## Irreducible error from the window grid",
        "",
        f"The annotated onset lies inside the window grid for "
        f"{cov['onset_inside_grid']:.3f} of test videos; videos whose onset falls "
        f"outside contribute at least {cov['mean_unreachable_error']:.3f} s on average.",
        "标注 onset 落在窗口网格内的比例，以及网格外样本贡献的不可约误差。",
        "",
    ]
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", default=str(HERE / "cache_emb_accident_full"))
    p.add_argument("--ckpt_root", default=str(HERE / "checkpoints/accident_full_t"))
    p.add_argument("--output_dir", default=str(HERE / "results/accident_full_t"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--phase_lr", type=float, default=3e-4)
    p.add_argument("--t_lr", type=float, default=1e-3)
    p.add_argument("--gru_hidden", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--blur_win", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_root).mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    root = Path(args.cache_root)
    mk = lambda ds, sh: DataLoader(  # noqa: E731
        ds, batch_size=args.batch_size, shuffle=sh,
        num_workers=args.num_workers, collate_fn=collate,
    )
    ds = {s: CrashEmbDataset(root / s) for s in ("train", "val", "test")}
    loaders = (mk(ds["train"], True), mk(ds["val"], False), mk(ds["test"], False))
    print(
        f"Device={device} train={len(ds['train'])} val={len(ds['val'])} "
        f"test={len(ds['test'])}"
    )

    res = {
        "n_train": len(ds["train"]),
        "n_val": len(ds["val"]),
        "n_test": len(ds["test"]),
        "coverage": coverage_report(loaders[2]),
    }
    print("coverage:", res["coverage"])

    print("\n===== phase head =====")
    phase_model, phase_test = train_phase_head(loaders, device, args)
    res["phase_only"] = phase_test
    # The phase evaluator names the median "median"; align it with the T metrics.
    # phase 评估器用 "median"，此处对齐 T 侧的键名。
    peak = phase_test["onset_from_impact_peak"]
    res["phase_only_onset"] = {**peak, "median_ae": peak["median"]}

    print("\n===== T head (emb only) =====")
    t_head, t_test = train_t_head(loaders, device, args, tag="t_only")
    res["t_only"] = t_test

    print("\n===== T head (concat predicted phase) =====")
    _, cat_test = train_t_head(
        loaders, device, args, phase_model=phase_model, use_phase=True, tag="phase_cat_T"
    )
    res["phase_cat_T"] = cat_test

    print("\n===== decode-time fusion =====")
    val_recs = t_head_scores(t_head, None, loaders[1], device, False)
    test_recs = t_head_scores(t_head, None, loaders[2], device, False)
    res["phase_prior_T"] = sweep_fusion(
        val_recs,
        test_recs,
        phase_prob_cache(phase_model, loaders[1], device),
        phase_prob_cache(phase_model, loaders[2], device),
        weights=[0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0],
        blur_win=args.blur_win,
    )
    print(
        f"TEST [phase_prior_T] w={res['phase_prior_T']['best_w']} "
        f"MAE={res['phase_prior_T']['test']['mae']:.3f} "
        f"T={res['phase_prior_T']['test']['t_score']:.3f}"
    )

    (out_dir / "0728_accident_full_t.json").write_text(json.dumps(res, indent=2))
    write_report(res, out_dir / "0728_accident_full_t.md")


if __name__ == "__main__":
    main()
