#!/usr/bin/env python3
"""Phase label helpers for Task-1 cumulative / 4-class heads.

Onset-derived phase boundaries (ACCIDENT matched protocol):
  normal:     center_time < onset - 2.0 s
  pre-crash:  onset - 2.0 s <= center_time < onset - 0.5 s
  impact:     abs(center_time - onset) <= 0.5 s
  post-crash: center_time > onset + 0.5 s

中文：阶段标签由 onset 时刻派生；pre-crash 安全区从 onset 前 2.0 s 开始。
"""

from __future__ import annotations

from typing import Literal

PhaseName = Literal["normal", "pre_impact", "impact", "post_impact"]

PHASE2ID = {"normal": 0, "pre_impact": 1, "impact": 2, "post_impact": 3}
ID2PHASE = {v: k for k, v in PHASE2ID.items()}
PHASE_ORDER = ("normal", "pre_impact", "impact", "post_impact")

# Seconds relative to onset. / 相对 onset 的秒数边界。
PRE_START = -2.0
IMPACT_HALF = 0.5
# Clean-normal negatives must end at least this far before onset.
# 负样本片段的最后窗口中心须不晚于 onset - CLEAN_END_MARGIN。
CLEAN_END_MARGIN = 3.0

# Legal adjacent phase transitions for the transition-consistency metric.
LEGAL_TRANSITIONS = frozenset(
    {
        (0, 0),  # normal -> normal
        (0, 1),  # normal -> pre
        (1, 1),  # pre -> pre
        (1, 2),  # pre -> impact
        (2, 2),  # impact -> impact
        (2, 3),  # impact -> post
        (3, 3),  # post -> post
    }
)


def assign_phase(t: float, t_gt: float | None) -> PhaseName:
    """Assign a 4-way phase from window center time and onset.

    根据窗口中心时刻与 onset 分配四阶段标签。
    """
    if t_gt is None:
        return "normal"
    if t < t_gt + PRE_START:  # < onset - 2.0
        return "normal"
    if t < t_gt - IMPACT_HALF:  # [onset-2.0, onset-0.5)
        return "pre_impact"
    if abs(t - t_gt) <= IMPACT_HALF:  # [onset-0.5, onset+0.5]
        return "impact"
    return "post_impact"  # > onset + 0.5


def phase_to_cumulative(phase: str) -> tuple[float | None, float | None, bool]:
    """Return (y_a, y_b, valid_for_phase_loss).

    impact -> ignored in auxiliary phase losses (valid=False).
    """
    if phase == "normal":
        return 0.0, 0.0, True
    if phase == "pre_impact":
        return 1.0, 0.0, True
    if phase == "impact":
        return None, None, False
    if phase == "post_impact":
        return 1.0, 1.0, True
    raise ValueError(f"unknown phase: {phase}")


def phase_ids_from_cum(y_a, y_b, phase_valid, valid=None):
    """Recover 4-class phase ids from the cached cumulative labels.

    Cached `phase_valid=False` marks impact windows (excluded from the cumulative
    a/b losses); with a 4-class head impact becomes a supervised class instead.
    Padding (valid=False) stays unlabeled and is returned as -1.
    """
    import numpy as np

    y_a = np.asarray(y_a)
    y_b = np.asarray(y_b)
    phase_valid = np.asarray(phase_valid, dtype=bool)
    valid = np.ones(len(y_a), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)

    ids = np.full(len(y_a), -1, dtype=np.int64)
    ids[valid] = PHASE2ID["impact"]  # default for valid-but-not-phase_valid
    lab = valid & phase_valid
    ids[lab & (y_a < 0.5) & (y_b < 0.5)] = PHASE2ID["normal"]
    ids[lab & (y_a >= 0.5) & (y_b < 0.5)] = PHASE2ID["pre_impact"]
    ids[lab & (y_a >= 0.5) & (y_b >= 0.5)] = PHASE2ID["post_impact"]
    return ids


def window_specs(n_frames: int, fps: float, window_sec: float = 2.0, stride_sec: float = 0.25):
    half = max(1, int(fps * window_sec / 2))
    stride = max(1, int(fps * stride_sec))
    specs = []
    c = half
    while c + half < n_frames:
        specs.append((c - half, c + half, c / fps))
        c += stride
    return specs
