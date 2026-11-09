#!/usr/bin/env python3
# 0728_train_phase_progression.py
"""E2: retrain the phase head for temporal coherence, as a minimal ablation ladder.

Why / 动机 (from 0728_phase_t_diagnostics.py):
  E1 showed the T head tolerates a blurred phase (±2 s smoothing is free) and even
  a several-second global offset, but collapses once per-window labels become
  inconsistent. Synthetic phase matching the measured confusion of the current
  model gained only -0.38 s. So the target is **temporal coherence**, not
  per-window accuracy.
  E1 表明：边界模糊几乎无损、整体偏移仍有用，但逐窗不一致会摧毁收益。
  因此重训目标是时间一致性（成块结构），而不是逐窗分类准确率。

Ablation ladder — each rung adds exactly one change, checkpoints kept.
消融阶梯 —— 每级只加一个改动，逐级保留 checkpoint。
  v0_baseline_matched  frozen Stage-I head (matched protocol, L=8 clips)
                       现有 Stage-I 头（matched 协议，L=8 片段），仅评估不训练
  v1_crashonly_long    crash-only data + full-length sequences, same 4-way softmax
                       仅 crash 视频 + 全长序列，头部结构不变
  v2_ordered           ordinal head: P(>=pre) >= P(>=impact) >= P(>=post)
                       有序累积头：同一时刻的类别顺序被强制
  v3_constrained       v2 + monotone-in-time progression (non-negative increments)
                       v2 + 时间单调进程（非负增量累加），非法回退不可能发生

Gate / 闸门: do NOT wire phase into the T head until impact-boundary and
temporal-consistency metrics improve substantially over v0.
在 impact 边界与时间一致性指标较 v0 有实质改善前，不接 T 头。
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
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labels import IMPACT_HALF, LEGAL_TRANSITIONS, PHASE_ORDER  # noqa: E402
from losses import balanced_bce_with_logits, balanced_ce_with_logits  # noqa: E402
from t_stage import (  # noqa: E402
    CrashEmbDataset,
    collate,
    cumulative_from_prob,
    load_stage1_phase_model,
    phase_ids_from_times,
    t_score,
)

HEAD_CHOICES = ("softmax4", "ordinal", "monotone")


# --------------------------------------------------------------------------
# model / 模型
# --------------------------------------------------------------------------
class PhaseProgressionModel(nn.Module):
    """BiGRU over frozen VideoMAE embeddings → 4-phase probabilities.

    Three head parameterizations share one encoder so the ablation isolates the
    head alone. / 三种输出头共用同一编码器，使消融只反映头部差异。

    softmax4  independent 4-way softmax per window (baseline parameterization)
              逐窗独立四分类，相邻窗之间没有任何约束
    ordinal   cumulative thresholds on a single scalar `raw_t`:
              P(phase >= k)_t = sigmoid(raw_t - theta_k), theta increasing.
              Guarantees class ordering *within* a timestep but `raw_t` may
              still jump around in time.
              单标量 + 递增阈值：保证同一时刻类别有序，但 raw_t 仍可随时间乱跳。
    monotone  same thresholds, but `raw_t` is built from non-negative increments
              so it never decreases in time. The predicted phase is then a
              staircase normal -> pre -> impact -> post that can never go back.
              raw_t 由非负增量累加而成，随时间单调不减，
              预测阶段必然是不可回退的阶梯，非法转移在结构上就不可能出现。
    """

    def __init__(
        self,
        emb_dim: int = 768,
        gru_hidden: int = 256,
        dropout: float = 0.3,
        head: str = "softmax4",
        bidirectional: bool = True,
    ):
        super().__init__()
        assert head in HEAD_CHOICES
        self.head = head
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        h_dim = gru_hidden * (2 if bidirectional else 1)
        self.drop = nn.Dropout(dropout)
        if head == "softmax4":
            self.out = nn.Linear(h_dim, 4)
        else:
            self.out = nn.Linear(h_dim, 1)
            # Ordered thresholds theta_1 <= theta_2 <= theta_3, kept ordered by
            # construction via positive gaps. / 用正间隔保证阈值递增。
            self.theta_base = nn.Parameter(torch.tensor(-1.0))
            self.theta_gap = nn.Parameter(torch.zeros(2))  # softplus(0) ~ 0.69
            # Starting level of the progression before any increment.
            # 进程起点（尚未累加任何增量时的水平）。
            self.raw_start = nn.Parameter(torch.tensor(-4.0))

    def thresholds(self) -> torch.Tensor:
        gaps = F.softplus(self.theta_gap)
        t1 = self.theta_base
        t2 = t1 + gaps[0]
        t3 = t2 + gaps[1]
        return torch.stack([t1, t2, t3])

    def encode(self, emb: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        emb = emb * valid_mask.unsqueeze(-1).float()
        lengths = valid_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=emb.size(1)
        )
        return self.drop(h)

    def forward(self, emb: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(emb, valid_mask)
        if self.head == "softmax4":
            logits4 = self.out(h)
            return {"logits4": logits4, "phase_prob": torch.softmax(logits4, dim=-1)}

        raw = self.out(h).squeeze(-1)  # [B,T]
        if self.head == "monotone":
            # Non-negative increments, zeroed on padding, then cumulative sum.
            # 非负增量（padding 处置零）后做前缀和 → 随时间单调不减。
            delta = F.softplus(raw) * valid_mask.float()
            raw = self.raw_start + torch.cumsum(delta, dim=1)
        raw = raw.clamp(-15.0, 15.0)

        cum_logits = raw.unsqueeze(-1) - self.thresholds().view(1, 1, 3)  # [B,T,3]
        p = torch.sigmoid(cum_logits)
        # Differences of the ordered cumulative probabilities give the 4 classes.
        # 有序累积概率作差得到四个类别的概率。
        phase_prob = torch.stack(
            [1.0 - p[..., 0], p[..., 0] - p[..., 1], p[..., 1] - p[..., 2], p[..., 2]],
            dim=-1,
        ).clamp(min=1e-6)
        return {"raw": raw, "cum_logits": cum_logits, "phase_prob": phase_prob}


def phase_loss(model: PhaseProgressionModel, out: dict, y_ids: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """Class-balanced supervision, matched to the head parameterization.

    两种头都用类别平衡损失，保证消融之间可比。
    """
    if model.head == "softmax4":
        return balanced_ce_with_logits(out["logits4"], y_ids.clamp(min=0), mask)
    # Ordinal: three balanced binary problems "is the phase at least k?".
    # 有序头：三个平衡的二分类「阶段是否已达到 k」。
    terms = []
    for k in range(3):
        target = (y_ids >= (k + 1)).float()
        terms.append(balanced_bce_with_logits(out["cum_logits"][..., k], target, mask))
    return torch.stack(terms).mean()


# --------------------------------------------------------------------------
# evaluation: impact boundary + temporal consistency
# --------------------------------------------------------------------------
def longest_run(ids: np.ndarray, value: int) -> tuple[int, int] | None:
    """Longest maximal run of `value`; returns (start_idx, end_idx) inclusive.

    返回最长的连续 value 段（闭区间下标），没有则 None。
    """
    best = None
    best_len = 0
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] != value:
            i += 1
            continue
        j = i
        while j + 1 < n and ids[j + 1] == value:
            j += 1
        if j - i + 1 > best_len:
            best_len = j - i + 1
            best = (i, j)
        i = j + 1
    return best


def interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Temporal IoU of two closed intervals. / 两个时间区间的 IoU。"""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return float(inter / union) if union > 0 else 0.0


def _stat(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return {"mean": None, "median": None}
    return {"mean": float(x.mean()), "median": float(np.median(x))}


def _err_stat(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    return {
        "mae": float(x.mean()),
        "median": float(np.median(x)),
        "within_1s": float((x <= 1.0).mean()),
        "within_2s": float((x <= 2.0).mean()),
        "t_score": float(np.mean([t_score(float(e)) for e in x])),
    }


def first_crossing_time(p: torch.Tensor, times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """First time where p crosses 0.5; falls back to the last valid window.

    p 首次越过 0.5 的时刻；若始终未越过则退化为最后一个有效窗。
    每个累积通道只有一个 0.5 交点，因此不受窄类被 argmax 挤掉的影响。
    """
    hit = (p >= 0.5) & mask
    has = hit.any(dim=1)
    idx = torch.where(has, hit.float().argmax(dim=1), (mask.sum(dim=1) - 1).clamp(min=0))
    return times.gather(1, idx.unsqueeze(1)).squeeze(1)


def _sequence_metrics(seq: np.ndarray, ts: np.ndarray, onset: float) -> dict:
    """Impact-interval and coherence stats for one decoded phase sequence.

    单条解码序列的 impact 区间与时间一致性统计。
    """
    out: dict = {}
    run = longest_run(seq, 2)
    out["detected"] = 1.0 if run is not None else 0.0
    if run is not None:
        pi = (float(ts[run[0]]), float(ts[run[1]]))
        gi = (onset - IMPACT_HALF, onset + IMPACT_HALF)
        out["start_err"] = abs(pi[0] - gi[0])
        out["end_err"] = abs(pi[1] - gi[1])
        out["center_err"] = abs(0.5 * (pi[0] + pi[1]) - onset)
        out["iou"] = interval_iou(pi, gi)
    if len(seq) >= 2:
        pairs = list(zip(seq[:-1], seq[1:]))
        out["switches"] = float(sum(a != b for a, b in pairs))
        out["legal"] = float(
            np.mean([(int(a), int(b)) in LEGAL_TRANSITIONS for a, b in pairs])
        )
        out["monotone"] = float(np.mean([b >= a for a, b in pairs]))
    out["distinct"] = float(len(np.unique(seq)))
    return out


def _aggregate(per_video: list[dict], yt: np.ndarray, yp: np.ndarray) -> dict:
    def col(key):
        return [d[key] for d in per_video if key in d]

    per_class = f1_score(yt, yp, average=None, labels=[0, 1, 2, 3], zero_division=0)
    return {
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "per_class_f1": {p: float(v) for p, v in zip(PHASE_ORDER, per_class)},
        "pred_class_share": {p: float((yp == i).mean()) for i, p in enumerate(PHASE_ORDER)},
        "confusion_rows_gt": confusion_matrix(yt, yp, labels=[0, 1, 2, 3]).tolist(),
        "impact_boundary": {
            "detect_rate": float(np.mean(col("detected"))),
            "start_err": _stat(col("start_err")),
            "end_err": _stat(col("end_err")),
            "center_err": _stat(col("center_err")),
            "iou": _stat(col("iou")),
        },
        "temporal_consistency": {
            "switches_per_video": _stat(col("switches")),
            "legal_transition_rate": float(np.mean(col("legal"))) if col("legal") else None,
            "monotone_rate": float(np.mean(col("monotone"))) if col("monotone") else None,
            "distinct_phases": _stat(col("distinct")),
        },
    }


@torch.no_grad()
def evaluate_phase(prob_fn, loader, device) -> dict:
    """Impact-boundary and temporal-consistency metrics on full crash videos.

    `prob_fn(emb, mask) -> phase_prob [B,T,4]` so the baseline Stage-I model and
    the new heads are scored by exactly the same code.
    用同一套代码评估基线与新头部，保证可比。

    Two decoders are reported because argmax is structurally biased against the
    impact class: impact spans only +-0.5 s (~5 windows), so a neighbouring class
    can win the argmax everywhere even when P(impact) is well localized.
    报告两种解码：argmax 对 impact 这种窄类有结构性偏见（impact 仅 ±0.5s 约 5 窗），
    即使 P(impact) 定位很准，argmax 也可能全程被邻类占据。
      argmax — per-window maximum probability / 逐窗取最大概率
      cum50  — largest k with P(phase >= k) >= 0.5 / 满足累积概率过半的最大 k
    """
    decodes = ("argmax", "cum50")
    per_video = {d: [] for d in decodes}
    yt_all = []
    yp_all = {d: [] for d in decodes}
    peak_err, cross_err, impact_start_err, gt_switches = [], [], [], []

    for batch in loader:
        emb = batch["emb"].to(device)
        mask = batch["valid_mask"].to(device)
        times = batch["times"].to(device)
        onset = batch["t_gt"].to(device)
        prob = prob_fn(emb, mask)
        cum = cumulative_from_prob(prob)
        gt_ids = phase_ids_from_times(times, onset)
        m = mask.bool()

        ids = {
            "argmax": prob.argmax(dim=-1),
            "cum50": ((cum >= 0.5) & m.unsqueeze(-1)).sum(dim=-1),
        }
        yt_all.append(gt_ids[m].cpu().numpy())
        for d in decodes:
            yp_all[d].append(ids[d][m].cpu().numpy())

        # onset estimate 1: peak of P(impact) / 估计一：P(impact) 峰值
        p_imp = prob[..., 2].masked_fill(~m, -1.0)
        t_peak = times.gather(1, p_imp.argmax(dim=1, keepdim=True)).squeeze(1)
        peak_err.extend((t_peak - onset).abs().cpu().tolist())

        # onset estimate 2: midpoint of the two cumulative 0.5 crossings, since
        # impact runs from onset-0.5 s to onset+0.5 s by definition.
        # 估计二：两个累积 0.5 交点的中点（impact 区间按定义就是 onset±0.5s）。
        t_imp = first_crossing_time(cum[..., 1], times, m)
        t_post = first_crossing_time(cum[..., 2], times, m)
        cross_err.extend((0.5 * (t_imp + t_post) - onset).abs().cpu().tolist())
        impact_start_err.extend((t_imp - (onset - IMPACT_HALF)).abs().cpu().tolist())

        n_valid = mask.sum(dim=1).cpu().numpy()
        times_np = times.cpu().numpy()
        gt_np = gt_ids.cpu().numpy()
        onset_np = onset.cpu().numpy()
        ids_np = {d: ids[d].cpu().numpy() for d in decodes}
        for b in range(emb.size(0)):
            L = int(n_valid[b])
            for d in decodes:
                per_video[d].append(
                    _sequence_metrics(ids_np[d][b, :L], times_np[b, :L], float(onset_np[b]))
                )
            g = gt_np[b, :L]
            if L >= 2:
                gt_switches.append(float(sum(a != b_ for a, b_ in zip(g[:-1], g[1:]))))

    yt = np.concatenate(yt_all)
    result = {
        "n_videos": int(len(peak_err)),
        "n_windows": int(len(yt)),
        "onset_from_impact_peak": _err_stat(peak_err),
        "onset_from_crossings": _err_stat(cross_err),
        "impact_start_err": _stat(impact_start_err),
        "gt_switches_per_video": _stat(gt_switches),
        "decode": {
            d: _aggregate(per_video[d], yt, np.concatenate(yp_all[d])) for d in decodes
        },
    }
    # Single number for model selection and for the gate: the better of the two
    # onset estimators. / 用于选型与闸门的单一指标：两种 onset 估计中较好者。
    result["best_onset_mae"] = min(
        result["onset_from_impact_peak"]["mae"], result["onset_from_crossings"]["mae"]
    )
    result["best_impact_f1"] = max(
        result["decode"][d]["per_class_f1"]["impact"] for d in decodes
    )
    return result


# --------------------------------------------------------------------------
# training / 训练
# --------------------------------------------------------------------------
def train_variant(tag: str, head: str, loaders, device, args) -> dict:
    train_loader, val_loader, test_loader = loaders
    out_dir = Path(args.ckpt_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_dim = next(iter(train_loader))["emb"].shape[-1]
    model = PhaseProgressionModel(
        emb_dim=emb_dim,
        gru_hidden=args.gru_hidden,
        dropout=args.dropout,
        head=head,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def prob_fn(emb, mask):
        model.eval()
        return model(emb, mask)["phase_prob"]

    # Selection metric is the onset error implied by the phase head itself:
    # it is the quantity the downstream T stage would actually consume.
    # 选择指标用 phase 自身隐含的 onset 误差 —— 这正是下游 T 阶段真正要用的量。
    best_sel, bad, history = 1e9, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"{tag} ep{epoch}", leave=False):
            emb = batch["emb"].to(device)
            mask = batch["valid_mask"].to(device)
            y_ids = phase_ids_from_times(batch["times"].to(device), batch["t_gt"].to(device))
            out = model(emb, mask)
            loss = phase_loss(model, out, y_ids, mask)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            losses.append(float(loss))
        sched.step()

        val = evaluate_phase(prob_fn, val_loader, device)
        sel = val["best_onset_mae"]
        sw = val["decode"]["cum50"]["temporal_consistency"]["switches_per_video"]["median"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "val_onset_mae": sel,
                "val_impact_f1": val["best_impact_f1"],
                "val_switches_cum50": sw,
            }
        )
        print(
            f"[{tag} {epoch}] loss={np.mean(losses):.4f} "
            f"val_onsetMAE={sel:.3f} impactF1={val['best_impact_f1']:.3f} "
            f"switches_cum50(med)={sw:.0f}"
        )
        if sel < best_sel - 1e-4:
            best_sel, bad = sel, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "head": head,
                    "tag": tag,
                    "epoch": epoch,
                    "emb_dim": emb_dim,
                    "val": val,
                    "args": vars(args),
                },
                out_dir / "best.pt",
            )
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[{tag}] early stop at epoch {epoch}")
                break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate_phase(prob_fn, test_loader, device)
    print(
        f"TEST [{tag}] onsetMAE={test['best_onset_mae']:.3f} "
        f"(peak={test['onset_from_impact_peak']['mae']:.3f} "
        f"cross={test['onset_from_crossings']['mae']:.3f}) "
        f"impactF1={test['best_impact_f1']:.3f} "
        f"switches_cum50(med)="
        f"{test['decode']['cum50']['temporal_consistency']['switches_per_video']['median']:.0f}"
    )
    return {
        "tag": tag,
        "head": head,
        "trained": True,
        "best_epoch": int(ckpt["epoch"]),
        "ckpt": str(out_dir / "best.pt"),
        "val": ckpt["val"],
        "test": test,
    }


def eval_baseline(args, loaders, device) -> dict:
    """Score the existing Stage-I head with the identical metric code.

    用完全相同的指标代码评估现有 Stage-I 头，作为消融阶梯的第 0 级。
    """
    _, val_loader, test_loader = loaders
    base = load_stage1_phase_model(Path(args.baseline_ckpt), device)

    def prob_fn(emb, mask):
        return base(emb, mask)["phase_prob"]

    val = evaluate_phase(prob_fn, val_loader, device)
    test = evaluate_phase(prob_fn, test_loader, device)
    print(
        f"TEST [v0_baseline_matched] onsetMAE={test['best_onset_mae']:.3f} "
        f"impactF1={test['best_impact_f1']:.3f} "
        f"switches_cum50(med)="
        f"{test['decode']['cum50']['temporal_consistency']['switches_per_video']['median']:.0f}"
    )
    return {
        "tag": "v0_baseline_matched",
        "head": "softmax4",
        "trained": False,
        "ckpt": args.baseline_ckpt,
        "val": val,
        "test": test,
    }


# --------------------------------------------------------------------------
EMB_ONLY_T_MAE = 4.128  # emb-only T baseline from 0728_phase_t_diagnostics / T 基线


def gate_verdict(base: dict, best: dict) -> tuple[bool, list[str], list[str]]:
    """Decide whether phase may now be wired into the T head.

    判断 phase 是否可以接入 T 头。
    The gate covers exactly the two families that E1 validated as decisive:
    impact-boundary localization and temporal coherence. Per-window class
    accuracy is deliberately *not* a gate criterion — E1 showed blurred phase
    costs nothing while incoherent phase destroys the benefit, so per-window F1
    is the wrong target. It is reported as context only.
    闸门只看 E1 验证过的两族指标：impact 边界定位与时间一致性。
    逐窗分类准确率刻意不作为通过条件——E1 已证明模糊无害、不一致致命，
    因此逐窗 F1 是错误的优化目标，此处仅作参考。
    """
    bt, tt = base["test"], best["test"]
    b_dec, t_dec = bt["decode"]["cum50"], tt["decode"]["cum50"]
    checks, context = [], []

    d_onset = bt["best_onset_mae"] - tt["best_onset_mae"]
    ok_onset = d_onset >= 1.0 and tt["best_onset_mae"] < EMB_ONLY_T_MAE
    checks.append(
        f"impact boundary — onset MAE {bt['best_onset_mae']:.3f}s → "
        f"{tt['best_onset_mae']:.3f}s ({d_onset:+.3f}s, T baseline "
        f"{EMB_ONLY_T_MAE:.3f}s): {'PASS' if ok_onset else 'FAIL'}"
    )

    b_start = bt["impact_start_err"]["median"]
    t_start = tt["impact_start_err"]["median"]
    ok_start = (b_start - t_start) >= 1.0
    checks.append(
        f"impact boundary — impact start error {b_start:.3f}s → {t_start:.3f}s "
        f"({t_start - b_start:+.3f}s): {'PASS' if ok_start else 'FAIL'}"
    )

    det = t_dec["impact_boundary"]["detect_rate"]
    ok_det = det >= 0.90
    checks.append(
        f"impact boundary — impact segment detected on {det:.3f} of videos "
        f"(baseline {b_dec['impact_boundary']['detect_rate']:.3f}): "
        f"{'PASS' if ok_det else 'FAIL'}"
    )

    sw = t_dec["temporal_consistency"]["switches_per_video"]["median"]
    gt_sw = tt["gt_switches_per_video"]["median"] or 3.0
    b_sw = b_dec["temporal_consistency"]["switches_per_video"]["median"]
    ok_sw = sw is not None and abs(sw - gt_sw) <= 2
    checks.append(
        f"temporal consistency — switches per video {b_sw:.0f} → {sw:.0f} "
        f"(ground truth {gt_sw:.0f}): {'PASS' if ok_sw else 'FAIL'}"
    )

    legal = t_dec["temporal_consistency"]["legal_transition_rate"]
    ok_legal = legal >= 0.99
    checks.append(
        f"temporal consistency — legal transitions "
        f"{b_dec['temporal_consistency']['legal_transition_rate']:.3f} → "
        f"{legal:.3f}: {'PASS' if ok_legal else 'FAIL'}"
    )

    context.append(
        f"per-window impact F1 {bt['best_impact_f1']:.3f} → {tt['best_impact_f1']:.3f} "
        "(context only; impact spans ~5 of ~121 windows, so this metric is "
        "dominated by class width rather than localization quality)"
    )
    context.append(
        f"impact interval IoU {b_dec['impact_boundary']['iou']['mean']:.3f} → "
        f"{t_dec['impact_boundary']['iou']['mean']:.3f} (context only; a 1 s "
        "ground-truth interval makes IoU harsh at 0.2 s window resolution)"
    )
    passed = all([ok_onset, ok_start, ok_det, ok_sw, ok_legal])
    return passed, checks, context


def write_report(results: list[dict], out_path: Path) -> None:
    base = next(r for r in results if r["tag"] == "v0_baseline_matched")
    trained = [r for r in results if r["trained"]]
    best = min(trained or [base], key=lambda r: r["test"]["best_onset_mae"])
    passed, checks, context = gate_verdict(base, best)

    lines = [
        "# E2 — phase retrained for temporal coherence (ablation ladder)",
        "",
        "Crash-only ACCIDENT, matched YouTube source split, full-length videos.",
        "Each rung adds exactly one change; every checkpoint is kept.",
        "每级只增加一个改动，所有 checkpoint 均保留。",
        "",
        "| Rung | Change | Head |",
        "|---|---|---|",
        "| v0_baseline_matched | existing Stage-I head, matched protocol, L=8 clips | softmax4 |",
        "| v1_crashonly_long | + crash-only data, + full-length sequences | softmax4 |",
        "| v2_ordered | + ordinal class ordering within each timestep | ordinal |",
        "| v3_constrained | + monotone progression in time (no going back) | monotone |",
        "",
        "## Onset localization from the phase head alone (test)",
        "",
        "`peak` = time of maximum P(impact); `crossings` = midpoint of the two",
        "cumulative 0.5 crossings, since impact spans onset ± 0.5 s by definition.",
        "For reference the emb-only T baseline is MAE 4.128 s.",
        "参考：emb-only T 基线 MAE 4.128 s。",
        "",
        "| Rung | onset MAE (peak) | onset MAE (crossings) | best | ≤1s | impact start err |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        t = r["test"]
        pk, cr = t["onset_from_impact_peak"], t["onset_from_crossings"]
        better = pk if pk["mae"] <= cr["mae"] else cr
        lines.append(
            f"| {r['tag']} | {pk['mae']:.3f} s | {cr['mae']:.3f} s | "
            f"**{t['best_onset_mae']:.3f} s** | {better['within_1s']:.3f} | "
            f"{t['impact_start_err']['median']:.3f} s |"
        )

    lines += [
        "",
        "## Impact boundary and coherence, per decoder (test)",
        "",
        "A ground-truth crash video is a clean staircase with about 3 switches.",
        "真实 crash 视频是一条约有 3 次切换的干净阶梯。",
        "",
        "| Rung | decode | impact F1 | detect | IoU | switches (med) | legal | monotone |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        for d, dd in r["test"]["decode"].items():
            ib, tc = dd["impact_boundary"], dd["temporal_consistency"]
            iou = ib["iou"]["mean"]
            lines.append(
                f"| {r['tag']} | {d} | {dd['per_class_f1']['impact']:.3f} | "
                f"{ib['detect_rate']:.3f} | "
                f"{'—' if iou is None else f'{iou:.3f}'} | "
                f"{tc['switches_per_video']['median']:.0f} | "
                f"{tc['legal_transition_rate']:.3f} | {tc['monotone_rate']:.3f} |"
            )
    gt_sw = results[0]["test"]["gt_switches_per_video"]["median"]
    lines += ["", f"Ground-truth switches per video (median): **{gt_sw:.0f}**", ""]

    lines += [
        "",
        "## Standalone onset localization vs the dedicated T stage",
        "",
        "Same crash videos, same split, same frozen encoder — only the",
        "supervision format differs. Numbers below come from",
        "`results/accident_t_complete/0727_accident_t_complete.md`.",
        "同数据、同 split、同冻结编码器，差别只在监督形式。",
        "",
        "| Method | Supervision | Test onset MAE |",
        "|---|---|---:|",
        "| A1 VideoMAE-head fine-tuned | Gaussian soft onset | 4.008 s |",
        "| B0 emb-only T head | Gaussian soft onset | 3.951 s |",
        f"| {best['tag']} (phase head, no T stage) | 4-phase progression | "
        f"**{best['test']['best_onset_mae']:.3f} s** |",
        "",
        "## Gate — may phase be wired into the T head?",
        "",
        f"Best rung by onset MAE: **{best['tag']}**",
        "",
        "Gate criteria / 闸门条件: impact boundary and temporal consistency,",
        "the two families E1 identified as decisive.",
        "",
    ]
    lines += [f"- {c}" for c in checks]
    lines += ["", "Context, not gate criteria / 参考指标，不作为通过条件:", ""]
    lines += [f"- {c}" for c in context]
    lines += [
        "",
        f"**Verdict: {'PROCEED to the T head' if passed else 'DO NOT wire into the T head yet'}.**",
        "",
    ]
    if not passed:
        lines += [
            "At least one impact-boundary or temporal-consistency criterion is",
            "unmet, so the phase output stays disconnected from the T stage.",
            "impact 边界或时间一致性尚未达标，phase 暂不接入 T 阶段。",
            "",
        ]
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_root", default=str(HERE / "cache_emb_accident"))
    p.add_argument(
        "--baseline_ckpt",
        default=str(HERE / "checkpoints/accident_matched/phase_only/best.pt"),
    )
    p.add_argument(
        "--ckpt_root",
        default=str(HERE / "checkpoints/accident_crashonly_phase"),
    )
    p.add_argument("--output_dir", default=str(HERE / "results/phase_progression"))
    p.add_argument(
        "--rungs",
        default="v0_baseline_matched,v1_crashonly_long,v2_ordered,v3_constrained",
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--gru_hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--grad_clip", type=float, default=1.0)
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
    loaders = (
        mk(CrashEmbDataset(root / "train"), True),
        mk(CrashEmbDataset(root / "val"), False),
        mk(CrashEmbDataset(root / "test"), False),
    )
    print(
        f"Device={device} crash videos "
        f"train={len(loaders[0].dataset)} val={len(loaders[1].dataset)} "
        f"test={len(loaders[2].dataset)}"
    )

    ladder = {
        "v1_crashonly_long": "softmax4",
        "v2_ordered": "ordinal",
        "v3_constrained": "monotone",
    }
    wanted = [r.strip() for r in args.rungs.split(",") if r.strip()]
    json_path = out_dir / "0728_phase_progression.json"
    if not wanted:
        # Re-render the report from saved results without retraining.
        # 仅用已保存结果重新生成报告，不重训。
        write_report(json.loads(json_path.read_text()), out_dir / "0728_phase_progression.md")
        return

    results = []
    for tag in wanted:
        print(f"\n===== {tag} =====")
        if tag == "v0_baseline_matched":
            results.append(eval_baseline(args, loaders, device))
        else:
            results.append(train_variant(tag, ladder[tag], loaders, device, args))
        # Persist after every rung so a later failure cannot lose earlier work.
        # 每级跑完即落盘，避免后续失败丢失已有结果。
        json_path.write_text(json.dumps(results, indent=2))

    write_report(results, out_dir / "0728_phase_progression.md")


if __name__ == "__main__":
    main()
