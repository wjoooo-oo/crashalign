#!/usr/bin/env python3
"""Fixed-L matched pos/neg sampling from a full-video window sequence.

No padding: every returned clip has exactly L valid windows.
训练时正样本 onset 位置随机；验证/测试用稳定 hash 选择。
"""

from __future__ import annotations

import hashlib

import numpy as np

from labels import (
    CLEAN_END_MARGIN,
    PHASE2ID,
    assign_phase,
    phase_to_cumulative,
)


def positive_starts(
    times: np.ndarray,
    onset: float,
    L: int,
    window_sec: float = 2.0,
) -> list[int]:
    """Start indices whose L-window clip temporally covers the onset.

    返回所有能覆盖 onset 的长度为 L 的起始下标。
    """
    half = window_sec / 2.0
    T = len(times)
    starts = []
    for i in range(0, T - L + 1):
        if times[i] - half <= onset <= times[i + L - 1] + half:
            starts.append(i)
    return starts


def negative_starts(
    times: np.ndarray,
    onset: float,
    L: int,
    end_margin: float = CLEAN_END_MARGIN,
) -> list[int]:
    """Start indices for clean-normal clips ending >= end_margin before onset.

    负样本：最后窗口中心 <= onset - end_margin，且不进入 pre-crash 安全区。
    """
    T = len(times)
    starts = []
    for i in range(0, T - L + 1):
        if float(times[i + L - 1]) <= onset - end_margin:
            starts.append(i)
    return starts


def gap_pair_candidates(
    times: np.ndarray,
    onset: float,
    L: int,
    gap_sec: float,
    window_sec: float = 2.0,
    pre_margin: float = 1.0,
    post_margin: float = 1.0,
) -> dict[int, list[int]]:
    """Positive starts (onset well inside the clip) mapped to their gap-respecting negatives.

    Gap mode (0925 re-split): the positive keeps the onset at least `pre_margin`
    after its start and `post_margin` before its end; a negative is valid for that
    positive only if it ends >= `gap_sec` before the positive starts and stays clean
    (last center <= onset - CLEAN_END_MARGIN). Positives with no valid negative are
    dropped, so a video is usable iff the returned dict is non-empty.

    间隔模式（0925 重划分）：正样本 onset 距片段起点 ≥ pre_margin、距终点 ≥ post_margin；
    负样本须在该正样本开始前 ≥ gap_sec 秒结束且保持 clean。没有可用负样本的正样本起点被丢弃，
    返回字典非空即视频可用。
    """
    half = window_sec / 2.0
    T = len(times)
    clip_start = [float(times[i]) - half for i in range(T - L + 1)]
    clip_end = [float(times[i + L - 1]) + half for i in range(T - L + 1)]
    clean = [i for i in range(T - L + 1) if float(times[i + L - 1]) <= onset - CLEAN_END_MARGIN]
    pairs: dict[int, list[int]] = {}
    for p in range(T - L + 1):
        if not (clip_start[p] + pre_margin <= onset <= clip_end[p] - post_margin):
            continue
        negs = [n for n in clean if clip_end[n] <= clip_start[p] - gap_sec + 1e-6]
        if negs:
            pairs[p] = negs
    return pairs


def pick_start(
    starts: list[int],
    *,
    key: str,
    train: bool,
    rng: np.random.Generator,
) -> int:
    """Random start at train time; deterministic hash pick for val/test.

    训练随机；验证/测试用 md5(key) 做稳定选择。
    """
    if not starts:
        raise ValueError(f"no valid starts for key={key}")
    if train:
        return int(rng.choice(starts))
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return starts[int(h, 16) % len(starts)]


def phase_targets_for_clip(
    times_clip: np.ndarray,
    onset: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build y_a, y_b, phase_valid, y_phase for a clip.

    为 clip 生成累计标签与 4 类 phase id；负样本 onset=None → 全 normal。
    """
    L = len(times_clip)
    y_a = np.zeros(L, dtype=np.float32)
    y_b = np.zeros(L, dtype=np.float32)
    phase_valid = np.ones(L, dtype=np.bool_)
    y_phase = np.zeros(L, dtype=np.int64)
    for i, t in enumerate(times_clip):
        ph = assign_phase(float(t), onset)
        y_phase[i] = PHASE2ID[ph]
        ya, yb, valid = phase_to_cumulative(ph)
        phase_valid[i] = valid
        if valid:
            y_a[i] = float(ya)
            y_b[i] = float(yb)
    return y_a, y_b, phase_valid, y_phase


def sample_matched_clip(
    emb: np.ndarray,
    times: np.ndarray,
    *,
    onset: float,
    role: str,
    L: int,
    train: bool,
    rng: np.random.Generator,
    video_id: str,
    window_sec: float = 2.0,
    gap_sec: float | None = None,
) -> dict:
    """Sample one fixed-L clip (positive or clean-normal negative).

    With `gap_sec`, positive and negative are drawn as a pair so the negative always
    ends >= gap_sec before the positive actually used (see gap_pair_candidates).
    Without it, the original independent sampling is kept.

    从全长序列中采样一个定长 clip（正样本或同视频 clean-normal 负样本）。
    给定 gap_sec 时正负成对采样，保证负样本在实际正样本开始前 ≥ gap_sec 秒结束；
    否则保持原来的独立采样。
    """
    if gap_sec is not None:
        pairs = gap_pair_candidates(times, onset, L, gap_sec, window_sec=window_sec)
        pos_start = pick_start(
            sorted(pairs), key=f"{video_id}|positive|L={L}|gap={gap_sec}", train=train, rng=rng
        )
        neg_start = pick_start(
            pairs[pos_start], key=f"{video_id}|negative|L={L}|gap={gap_sec}", train=train, rng=rng
        )
        if role == "positive":
            starts, start, is_crash, label_onset = sorted(pairs), pos_start, 1, onset
        elif role == "negative":
            starts, start, is_crash, label_onset = pairs[pos_start], neg_start, 0, None
        else:
            raise ValueError(role)
        return _build_clip(emb, times, start, L, label_onset, is_crash, len(starts), role, onset)

    if role == "positive":
        starts = positive_starts(times, onset, L, window_sec=window_sec)
        key = f"{video_id}|positive|L={L}"
        is_crash = 1
        label_onset: float | None = onset
    elif role == "negative":
        starts = negative_starts(times, onset, L)
        key = f"{video_id}|negative|L={L}"
        is_crash = 0
        label_onset = None  # force all-normal labels / 负样本强制全 normal
    else:
        raise ValueError(role)

    start = pick_start(starts, key=key, train=train, rng=rng)
    return _build_clip(emb, times, start, L, label_onset, is_crash, len(starts), role, onset)


def _build_clip(
    emb: np.ndarray,
    times: np.ndarray,
    start: int,
    L: int,
    label_onset: float | None,
    is_crash: int,
    n_candidates: int,
    role: str,
    onset: float,
) -> dict:
    """Slice the L windows and attach labels. / 切出 L 个窗口并附上标签。"""
    sl = slice(start, start + L)
    times_c = times[sl].astype(np.float32)
    emb_c = emb[sl].astype(np.float32)
    y_a, y_b, phase_valid, y_phase = phase_targets_for_clip(times_c, label_onset)
    valid = np.ones(L, dtype=np.bool_)
    return {
        "emb": emb_c,
        "times": times_c,
        "y_a": y_a,
        "y_b": y_b,
        "phase_valid": phase_valid,
        "y_phase": y_phase,
        "valid_mask": valid,
        "is_crash": is_crash,
        "start_index": start,
        "n_candidate_starts": n_candidates,
        "role": role,
        "onset": onset,
    }
