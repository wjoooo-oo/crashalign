"""
Temporal frame sampling strategies for crash explanation.
事故解释用的时间帧采样策略（相对 sub5 预测 onset 的秒级偏移）。

pre-heavy: 碰撞前帧更多，捕捉接近动力学（文献常见做法）。
offsets 单位为秒，负数=碰撞前，0=onset，正数=碰撞后。
"""

from __future__ import annotations

# Seconds relative to predicted/GT onset.
# Pre-onset frames capture approach dynamics (often more informative for WHY).
FRAME_STRATEGIES: dict[str, dict] = {
    # Current v6 default: symmetric ±3s
    "sym9": {
        "offsets": [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0],
        "desc": "Symmetric ±3s, 9 frames (v6 baseline)",
    },
    # Pre-heavy: 8 before, 3 after — causal context before impact
    "pre11": {
        "offsets": [-5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0],
        "desc": "Pre-heavy 8+3, covers −5..+2s",
    },
    # Wider pre for onset error (±1–2s): 10 before, 3 after
    "pre13": {
        "offsets": [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
        "desc": "Wide pre −7..+2s, robust to early onset error",
    },
    # Literature-style hybrid (cf. multi-stage VLM): sparse early + dense around onset
    "hybrid12": {
        "offsets": [-8.0, -6.0, -4.0, -2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0],
        "desc": "Sparse pre-context + dense ±2s (12 frames)",
    },
    # Dense local window at 4fps equivalent ±2s (up to 17 frames) — Pass-2 style
    "dense17": {
        "offsets": [
            -2.0, -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25,
            0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0,
        ],
        "desc": "Dense ±2s at ~4fps (17 frames), onset-centered",
    },
    # pre13 + dense17 combo would be too many; use pre+dense hybrid
    "pre_dense15": {
        "offsets": [
            -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
            0.0, 0.25, 0.5, 0.75, 1.0, 1.5,
        ],
        "desc": "Sparse −6/−4/−3s + dense −2..+1.5s (15 frames)",
    },
    # --- Controlled comparisons: same frame count, symmetric vs pre-heavy ---
    "sym11": {
        "offsets": [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "desc": "Symmetric ±5s, 11 frames (control for pre11)",
    },
    "sym15": {
        "offsets": [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "desc": "Symmetric ±7s, 15 frames (control for pre13/pre_dense15)",
    },
    # --- More frames / wider pre ---
    "pre17": {
        "offsets": [
            -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
            0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
        ],
        "desc": "Wide pre −8..+3s, 17 frames (12 pre + 5 post)",
    },
    "pre20": {
        "offsets": [
            -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
            -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
        ],
        "desc": "Very wide pre −10..+4s, 20 frames (11 pre + 9 post/at)",
    },
    # Controlled pre/post allocation ablation:
    # 固定总数 20、窗口 [-10,+4]、包含 onset，只改变 pre/post 帧数。
    "alloc20_pre14": {
        "offsets": [
            -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
            -2.5, -2.0, -1.5, -1.0, -0.5, -0.25,
            0.0, 0.8, 1.6, 2.4, 3.2, 4.0,
        ],
        "desc": "Controlled [-10,+4]s, 20 frames: 14 pre + 6 onset/post",
    },
    "alloc20_pre17": {
        "offsets": [
            -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.5, -4.0, -3.5,
            -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
            0.0, 2.0, 4.0,
        ],
        "desc": "Controlled [-10,+4]s, 20 frames: 17 pre + 3 onset/post",
    },
    # Symmetric pre↔post swap at fixed 25 frames and [-12,+6]s support.
    # 固定 25 帧与 [-12,+6]s 窗口，仅交换 pre/post 配额，检验「pre 是否更重要」。
    "alloc25_pre15": {
        "offsets": [
            -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
            -2.5, -2.0, -1.5, -1.0, -0.5,
            0.0, 0.75, 1.5, 2.25, 3.0, 3.75, 4.5, 5.25, 5.75, 6.0,
        ],
        "desc": "Controlled [-12,+6]s, 25 frames: 15 pre + 10 onset/post",
    },
    "alloc25_pre10": {
        "offsets": [
            -12.0, -10.5, -9.0, -7.5, -6.0, -4.5, -3.0, -2.0, -1.0, -0.5,
            0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6,
            4.0, 4.5, 5.0, 5.5, 6.0,
        ],
        "desc": "Controlled [-12,+6]s, 25 frames: 10 pre + 15 onset/post",
    },
    "pre13_post7": {
        "offsets": [
            -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, -0.5,
            0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0,
        ],
        "desc": "20 frames: 13 pre-onset + 7 post-onset (no dedicated onset frame)",
    },
    "pre25": {
        "offsets": [
            -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0,
            -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
            3.0, 3.5, 4.0, 4.5, 5.0,
        ],
        "desc": "Wide pre −12..+5s, 25 frames (14 pre + 11 post/at)",
    },
    "pre30": {
        "offsets": [
            -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0,
            -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0,
            1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
        ],
        "desc": "Very wide pre −15..+6s, 30 frames (17 pre + 13 post/at)",
    },
    "hybrid17": {
        "offsets": [
            -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, -0.25,
            0.0, 0.25, 0.5, 0.75, 1.0, 1.5,
        ],
        "desc": "Literature hybrid: sparse −10..−4s + dense ±1.5s (17 frames)",
    },
}


def get_offsets(name: str) -> list[float]:
    if name not in FRAME_STRATEGIES:
        raise KeyError(f"Unknown strategy {name!r}. Choose from {list(FRAME_STRATEGIES)}")
    return list(FRAME_STRATEGIES[name]["offsets"])


def count_pre_post(offsets: list[float]) -> tuple[int, int]:
    pre = sum(1 for o in offsets if o < 0)
    post = sum(1 for o in offsets if o > 0)
    at = sum(1 for o in offsets if o == 0)
    return pre, post + at
