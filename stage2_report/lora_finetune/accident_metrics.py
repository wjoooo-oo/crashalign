#!/usr/bin/env python3
"""
0625_ACCIDENT benchmark TSC metrics (arXiv:2604.09819, Sec. 5).

Implements temporal (T), spatial (S), collision-type (C), and unified ACC^S score.
实现 ACCIDENT 论文中的 T / S / C 及统一 ACC^S 评分。

Competition / sub5 pipeline uses isotropic spatial Gaussian with sigma_space=0.1.
竞赛与 sub5 管线使用各向同性空间高斯，sigma_space=0.1。

Paper Eq. (3) uses anisotropic Gaussian with sigma_x, sigma_y from mean box size.
论文式 (3) 使用各向异性高斯，sigma_x/sigma_y 来自标注框平均宽高。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def temporal_score(t_gt: float, t_pred: float, sigma_t: float = 2.0) -> float:
    """
    Per-video temporal similarity (paper Eq. 1).
    单视频时间相似度（论文式 1）。
    """
    sigma_t = max(float(sigma_t), 1e-9)
    dt = float(t_gt) - float(t_pred)
    return math.exp(-(dt * dt) / (2.0 * sigma_t * sigma_t))


def spatial_score_isotropic(
    x_gt: float,
    y_gt: float,
    x_pred: float,
    y_pred: float,
    sigma: float = 0.1,
) -> float:
    """
    Isotropic spatial score: exp(-d^2 / (2 sigma^2)), d = Euclidean distance.
    各向同性空间分：欧氏距离高斯衰减（竞赛默认 sigma=0.1）。
    """
    sigma = max(float(sigma), 1e-9)
    dx = float(x_gt) - float(x_pred)
    dy = float(y_gt) - float(y_pred)
    d2 = dx * dx + dy * dy
    return math.exp(-d2 / (2.0 * sigma * sigma))


def spatial_score_anisotropic(
    x_gt: float,
    y_gt: float,
    x_pred: float,
    y_pred: float,
    sigma_x: float,
    sigma_y: float,
) -> float:
    """
    Anisotropic spatial score (paper Eq. 3).
    各向异性空间分（论文式 3）。
    """
    sigma_x = max(float(sigma_x), 1e-9)
    sigma_y = max(float(sigma_y), 1e-9)
    dx = float(x_gt) - float(x_pred)
    dy = float(y_gt) - float(y_pred)
    return math.exp(-((dx * dx) / (2.0 * sigma_x * sigma_x) + (dy * dy) / (2.0 * sigma_y * sigma_y)))


def type_score(type_gt: str, type_pred: str) -> float:
    """Top-1 collision type accuracy indicator (paper Eq. 5)."""
    return 1.0 if str(type_gt).strip().lower() == str(type_pred).strip().lower() else 0.0


def harmonic_mean_tsc(t: float, s: float, c: float) -> float:
    """
    Unified ACC^S score (paper Eq. 6): 3 / (1/T + 1/S + 1/C).
    统一 ACC^S 调和平均（论文式 6）。
    """
    t, s, c = float(t), float(s), float(c)
    if t <= 0.0 or s <= 0.0 or c <= 0.0:
        return 0.0
    return 3.0 / (1.0 / t + 1.0 / s + 1.0 / c)


def evaluate_submission_rows(
    rows: Iterable[dict],
    *,
    sigma_t: float = 2.0,
    sigma_space: float = 0.1,
    anisotropic: bool = False,
    default_sigma_x: float = 0.1,
    default_sigma_y: float = 0.1,
) -> dict[str, float]:
    """
    Score aligned prediction/GT rows.
    对对齐后的预测/GT 行计算 T/S/C/HM。

    Each row must contain gt_* and pred_* keys or standard CSV column names.
    """
    t_scores: list[float] = []
    s_scores: list[float] = []
    c_scores: list[float] = []
    dt_abs: list[float] = []

    for row in rows:
        t_gt = float(row.get("gt_accident_time", row.get("accident_time_gt", row.get("accident_time", 0))))
        t_pred = float(row.get("pred_accident_time", row.get("accident_time_pred", row.get("accident_time", 0))))
        x_gt = float(row.get("gt_center_x", row.get("center_x_gt", row.get("center_x", 0.5))))
        y_gt = float(row.get("gt_center_y", row.get("center_y_gt", row.get("center_y", 0.5))))
        x_pred = float(row.get("pred_center_x", row.get("center_x_pred", row.get("center_x", 0.5))))
        y_pred = float(row.get("pred_center_y", row.get("center_y_pred", row.get("center_y", 0.5))))
        type_gt = row.get("gt_type", row.get("type_gt", row.get("type", "")))
        type_pred = row.get("pred_type", row.get("type_pred", row.get("type", "")))

        t_scores.append(temporal_score(t_gt, t_pred, sigma_t))
        dt_abs.append(abs(t_gt - t_pred))

        if anisotropic:
            sx = float(row.get("sigma_x", default_sigma_x))
            sy = float(row.get("sigma_y", default_sigma_y))
            s_scores.append(spatial_score_anisotropic(x_gt, y_gt, x_pred, y_pred, sx, sy))
        else:
            s_scores.append(spatial_score_isotropic(x_gt, y_gt, x_pred, y_pred, sigma_space))

        c_scores.append(type_score(type_gt, type_pred))

    n = len(t_scores)
    t_mean = float(np.mean(t_scores)) if n else 0.0
    s_mean = float(np.mean(s_scores)) if n else 0.0
    c_mean = float(np.mean(c_scores)) if n else 0.0
    return {
        "T": round(t_mean, 4),
        "S": round(s_mean, 4),
        "C": round(c_mean, 4),
        "ACC_S": round(harmonic_mean_tsc(t_mean, s_mean, c_mean), 4),
        "n": n,
        "dt_mae": round(float(np.mean(dt_abs)), 4) if dt_abs else 0.0,
        "dt_gt2s_frac": round(float(np.mean([d > 2.0 for d in dt_abs])), 4) if dt_abs else 0.0,
        "sigma_t": sigma_t,
        "sigma_space": sigma_space,
        "anisotropic": anisotropic,
    }


def evaluate_submission_csv(
    submission_csv: Path | str,
    ground_truth_csv: Path | str,
    paths: list[str] | None = None,
    **kwargs,
) -> dict[str, float]:
    """
    Score submission CSV vs GT CSV, optionally restricted to `paths`.
    对 submission 与 GT CSV 评分，可限定 path 子集。
    """
    submission_csv = Path(submission_csv)
    ground_truth_csv = Path(ground_truth_csv)

    with submission_csv.open(encoding="utf-8") as f:
        sub = {r["path"]: r for r in csv.DictReader(f)}
    with ground_truth_csv.open(encoding="utf-8") as f:
        gt = {r["path"]: r for r in csv.DictReader(f)}

    eval_paths = paths if paths is not None else sorted(gt.keys())
    rows = []
    for path in eval_paths:
        if path not in gt:
            continue
        pred = sub.get(path, {})
        g = gt[path]
        rows.append(
            {
                "path": path,
                "gt_accident_time": g.get("accident_time", 0),
                "pred_accident_time": pred.get("accident_time", 0),
                "gt_center_x": g.get("center_x", 0.5),
                "gt_center_y": g.get("center_y", 0.5),
                "pred_center_x": pred.get("center_x", 0.5),
                "pred_center_y": pred.get("center_y", 0.5),
                "gt_type": g.get("type", ""),
                "pred_type": pred.get("type", ""),
            }
        )
    return evaluate_submission_rows(rows, **kwargs)
