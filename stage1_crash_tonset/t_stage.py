#!/usr/bin/env python3
"""Shared building blocks for the onset-localization (T) stage. Added 0729.

Library module, so it carries a plain importable name like `model.py` /
`labels.py` / `losses.py`; the creation date is recorded here instead of in the
filename because a `0729_`-prefixed module cannot be imported with a normal
`import` statement.
本文件是库模块，沿用 model.py / labels.py / losses.py 的无日期命名，
因为以数字开头的模块名无法用普通 import 语句导入；创建日期记录在此处。

Everything here used to live inside `0727_train_phase_cond_t.py` and was pulled
in by later scripts through `importlib`, which produced a four-deep chain of
script-importing-script. Collecting the shared pieces in one library removes
that chain.
这些组件原先散落在 0727_train_phase_cond_t.py 中，被后续脚本用 importlib 逐层引入，
形成了四层「脚本导入脚本」的链条；集中到一个库模块后该链条不再需要。

Conventions / 约定:
  - Onset targets are Gaussian soft scores over window centres, sigma = 2 s.
    onset 目标是以窗口中心为坐标的高斯软分数，sigma = 2 秒。
  - Onset is decoded by picking the peak score, matching the VideoMAE T recipe.
    onset 通过取分数峰值解码，与 VideoMAE 的 T 方案一致。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labels import IMPACT_HALF, PHASE2ID, PRE_START, assign_phase  # noqa: E402
from model import Task1CumPhaseModel  # noqa: E402

SIGMA_T = 2.0  # Gaussian soft-onset sigma in seconds / 高斯软 onset 的 sigma（秒）
N_PHASES = 4


# --------------------------------------------------------------------------
# metrics / 指标
# --------------------------------------------------------------------------
def t_score(err: float, sigma: float = SIGMA_T) -> float:
    """Gaussian tolerance score for one absolute onset error.

    单个 onset 绝对误差的高斯容忍分数。
    """
    return float(np.exp(-(err ** 2) / (2.0 * sigma ** 2)))


def onset_metrics(errors) -> dict:
    """Standard onset-localization summary over a list of absolute errors.

    onset 定位的标准汇总指标（输入为绝对误差列表）。
    """
    e = np.asarray(errors, dtype=np.float64)
    return {
        "n": int(e.size),
        "mae": float(e.mean()),
        "median_ae": float(np.median(e)),
        "within_1s": float((e <= 1.0).mean()),
        "within_2s": float((e <= 2.0).mean()),
        "t_score": float(np.mean([t_score(float(x)) for x in e])),
    }


def gaussian_targets(times: np.ndarray, t_gt: float, sigma: float = SIGMA_T) -> np.ndarray:
    """Soft onset scores in [0,1] over window centres. / 高斯软 onset 目标。"""
    return np.exp(-((times - t_gt) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


# --------------------------------------------------------------------------
# phase label helpers / 阶段标签工具
# --------------------------------------------------------------------------
def gt_phase_onehot(times: np.ndarray, t_gt: float) -> np.ndarray:
    """Per-window GT phase one-hot from the onset (numpy, single video).

    由 onset 派生的逐窗 GT 阶段 one-hot（numpy，单个视频）。
    """
    out = np.zeros((len(times), N_PHASES), dtype=np.float32)
    for i, t in enumerate(times):
        out[i, PHASE2ID[assign_phase(float(t), t_gt)]] = 1.0
    return out


def phase_ids_from_times(times: torch.Tensor, onset: torch.Tensor) -> torch.Tensor:
    """Vectorized GT phase ids from window centres and onset (batched).

    向量化的批量阶段标签。
    times [B,T], onset [B] -> ids [B,T] in {0:normal, 1:pre, 2:impact, 3:post}
    """
    d = times - onset.unsqueeze(1)
    ids = torch.full_like(d, 3.0)
    ids = torch.where(d <= IMPACT_HALF, torch.full_like(d, 2.0), ids)
    ids = torch.where(d < -IMPACT_HALF, torch.full_like(d, 1.0), ids)
    ids = torch.where(d < PRE_START, torch.zeros_like(d), ids)
    return ids.long()


def phase_onehot(ids: torch.Tensor) -> torch.Tensor:
    """ids [B,T] -> one-hot [B,T,4]."""
    return torch.nn.functional.one_hot(ids.clamp(0, N_PHASES - 1), num_classes=N_PHASES).float()


def cumulative_from_prob(prob: torch.Tensor) -> torch.Tensor:
    """4-class probabilities -> the 3 cumulative probabilities P(phase >= k).

    四类概率 -> 三个累积概率 P(阶段 >= k)。
    Lets heads with different parameterizations share one decoder, and avoids the
    argmax bias against `impact`, which spans only ~5 of ~120 windows.
    使不同参数化的头共用同一解码器，并避免 argmax 对窄类 impact 的偏见
    （impact 仅占约 120 个窗口中的 5 个）。
    """
    p_ge1 = 1.0 - prob[..., 0]
    p_ge2 = prob[..., 2] + prob[..., 3]
    p_ge3 = prob[..., 3]
    return torch.stack([p_ge1, p_ge2, p_ge3], dim=-1)


# --------------------------------------------------------------------------
# data / 数据
# --------------------------------------------------------------------------
class CrashEmbDataset(Dataset):
    """Full-video frozen embeddings of crash videos, for onset localization.

    事故视频的全长冻结表征，用于 onset 定位。每个样本是一整个视频，不做裁剪，
    因此训练与推理看到的序列长度分布一致。
    """

    def __init__(self, cache_dir: Path):
        self.files = sorted(Path(cache_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz in {cache_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        path = self.files[idx]
        with np.load(path, allow_pickle=True) as z:
            emb = z["emb"].astype(np.float32)
            times = z["times"].astype(np.float32)
            t_gt = float(z["t_gt"])
            if t_gt < 0:
                raise ValueError(f"missing t_gt in {path}")
            vid = str(z["video_id"]) if "video_id" in z.files else path.stem
        return {
            "emb": torch.from_numpy(emb),
            "times": torch.from_numpy(times),
            "y_onset": torch.from_numpy(gaussian_targets(times, t_gt)),
            "gt_phase": torch.from_numpy(gt_phase_onehot(times, t_gt)),
            "t_gt": torch.tensor(t_gt, dtype=torch.float32),
            "video_id": vid,
            "length": emb.shape[0],
        }


def collate(batch: list[dict]) -> dict:
    """Right-pad variable-length videos and return a boolean validity mask.

    对变长视频右侧补零，并返回有效位掩码。
    """
    B = len(batch)
    T = max(b["length"] for b in batch)
    D = batch[0]["emb"].shape[-1]
    emb = torch.zeros(B, T, D)
    times = torch.zeros(B, T)
    y = torch.zeros(B, T)
    gt_phase = torch.zeros(B, T, N_PHASES)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b["length"]
        emb[i, :L] = b["emb"]
        times[i, :L] = b["times"]
        y[i, :L] = b["y_onset"]
        gt_phase[i, :L] = b["gt_phase"]
        mask[i, :L] = True
    return {
        "emb": emb,
        "times": times,
        "y_onset": y,
        "gt_phase": gt_phase,
        "valid_mask": mask,
        "t_gt": torch.stack([b["t_gt"] for b in batch]),
        "video_id": [b["video_id"] for b in batch],
    }


# --------------------------------------------------------------------------
# model pieces / 模型组件
# --------------------------------------------------------------------------
class OnsetScoreHead(nn.Module):
    """Per-window MLP producing an onset soft-score logit.

    逐窗 MLP，输出 onset 软分数 logit。
    """

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,T,D] -> [B,T]
        return self.net(x).squeeze(-1)


def load_stage1_phase_model(ckpt_path: Path, device: torch.device) -> Task1CumPhaseModel:
    """Load a frozen Stage-I phase model in eval mode.

    加载冻结的 Stage-I 阶段模型（eval 模式，梯度关闭）。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    model = Task1CumPhaseModel(
        emb_dim=ckpt["emb_dim"],
        gru_hidden=args["gru_hidden"],
        dropout=0.0,
        crash_feat=args["crash_feat"],
        phase_mode="softmax4",
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """MSE averaged over valid windows only. / 仅在有效窗口上取平均的 MSE。"""
    diff = ((pred - target) ** 2) * mask.float()
    return diff.sum() / mask.float().sum().clamp(min=1.0)


@torch.no_grad()
def decode_onset(scores: torch.Tensor, times: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Peak soft-score -> onset time. / 软分数峰值解码为 onset 时刻。"""
    neg = torch.finfo(scores.dtype).min
    s = scores.masked_fill(~mask, neg)
    idx = s.argmax(dim=1)
    return times.gather(1, idx.unsqueeze(1)).squeeze(1)


# --------------------------------------------------------------------------
# phase conditioning for the T head / T 头的 phase 条件输入
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_phase_prob(
    phase_model: Task1CumPhaseModel | None,
    emb: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    gt_phase: torch.Tensor,
) -> torch.Tensor | None:
    """Phase features fed to the T head under one conditioning mode.

    按条件模式返回喂给 T 头的 phase 特征。
      emb_only     -> None (no phase channel) / 不加 phase 通道
      phase_oracle -> GT one-hot (upper bound, leaks the answer)
                      GT one-hot（上界，会泄漏答案）
      otherwise    -> frozen predicted phase posterior / 冻结的预测阶段后验
    """
    if mode == "emb_only":
        return None
    if mode == "phase_oracle":
        return gt_phase
    assert phase_model is not None
    return phase_model(emb, mask)["phase_prob"].detach()


def build_t_input(emb: torch.Tensor, phase_feat: torch.Tensor | None) -> torch.Tensor:
    """Concatenate the phase channel onto the embeddings when present.

    存在 phase 通道时拼接到表征之后。
    """
    return emb if phase_feat is None else torch.cat([emb, phase_feat], dim=-1)


@torch.no_grad()
def evaluate_t_head(head, phase_model, loader, device, mode: str) -> dict:
    """Peak-decode onset for every video in `loader` and summarize the errors.

    对 loader 中每个视频取峰解码 onset 并汇总误差。
    """
    head.eval()
    errs = []
    for batch in loader:
        emb = batch["emb"].to(device)
        mask = batch["valid_mask"].to(device)
        times = batch["times"].to(device)
        t_gt = batch["t_gt"].to(device)
        feat = predict_phase_prob(phase_model, emb, mask, mode, batch["gt_phase"].to(device))
        probs = torch.sigmoid(head(build_t_input(emb, feat)))
        t_pred = decode_onset(probs, times, mask)
        errs.extend((t_pred - t_gt).abs().cpu().tolist())
    return onset_metrics(errs)
