#!/usr/bin/env python3
"""Cumulative pre/post targets for crash evolution (Task 1).

Non-crash: y_pre = y_post = 0 for all windows (zero-risk).
Crash: continuous cumulative structure relative to t_gt:
  y_pre  ramps 0→1 on [t_gt - pre_lead, t_gt]
  y_post ramps 0→1 on [t_gt + post_lag, t_gt + post_end]
  with y_post <= y_pre.
"""

from __future__ import annotations

import numpy as np


def window_specs(n_frames: int, fps: float, window_sec: float = 2.0, stride_sec: float = 0.25):
    half = max(1, int(fps * window_sec / 2))
    stride = max(1, int(fps * stride_sec))
    specs = []
    c = half
    while c + half < n_frames:
        specs.append((c - half, c + half, c / fps))
        c += stride
    return specs


def cumulative_targets(
    times: np.ndarray,
    t_gt: float | None,
    pre_lead: float = 3.0,
    post_lag: float = 0.5,
    post_end: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_pre, y_post) in [0,1], same length as times."""
    t = np.asarray(times, dtype=np.float64)
    if t_gt is None:
        z = np.zeros_like(t)
        return z, z.copy()

    # evolution has started: ramp through pre-crash into impact
    y_pre = np.clip((t - (t_gt - pre_lead)) / max(pre_lead, 1e-6), 0.0, 1.0)
    # post-crash accumulation after a short lag past impact
    span = max(post_end - post_lag, 1e-6)
    y_post = np.clip((t - (t_gt + post_lag)) / span, 0.0, 1.0)
    y_post = np.minimum(y_post, y_pre)
    return y_pre.astype(np.float32), y_post.astype(np.float32)
