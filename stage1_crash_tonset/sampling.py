#!/usr/bin/env python3
"""Fixed-length sequence sampling to block duration shortcuts.

ACCIDENT crash clips are systematically longer than SOTAD non-crash clips
(val: crash med ~116 windows vs non med ~76). Evaluating on full sequences
lets the model use length/domain cues. We force every video to a fixed number
of windows via random contiguous crop (train) or center crop (eval).
"""

from __future__ import annotations

import numpy as np


def crop_or_pad(
    emb: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    phase_valid: np.ndarray,
    times: np.ndarray,
    fixed_len: int,
    mode: str = "random",
    rng: np.random.Generator | None = None,
):
    """Return arrays of temporal length == fixed_len plus a boolean valid mask."""
    t = emb.shape[0]
    d = emb.shape[1]
    if t >= fixed_len:
        if mode == "center":
            start = (t - fixed_len) // 2
        elif mode == "random":
            rng = rng or np.random.default_rng()
            start = int(rng.integers(0, t - fixed_len + 1))
        else:
            raise ValueError(mode)
        sl = slice(start, start + fixed_len)
        return (
            emb[sl],
            y_a[sl],
            y_b[sl],
            phase_valid[sl],
            times[sl],
            np.ones(fixed_len, dtype=bool),
        )
    # pad
    pad = fixed_len - t
    emb_o = np.zeros((fixed_len, d), dtype=emb.dtype)
    ya_o = np.zeros(fixed_len, dtype=y_a.dtype)
    yb_o = np.zeros(fixed_len, dtype=y_b.dtype)
    pv_o = np.zeros(fixed_len, dtype=bool)
    tm_o = np.zeros(fixed_len, dtype=times.dtype)
    valid = np.zeros(fixed_len, dtype=bool)
    emb_o[:t], ya_o[:t], yb_o[:t], pv_o[:t], tm_o[:t] = emb, y_a, y_b, phase_valid, times
    valid[:t] = True
    return emb_o, ya_o, yb_o, pv_o, tm_o, valid
