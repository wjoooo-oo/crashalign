"""Video path resolution and frame extraction."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crashalign_paths import accident_root  # noqa: E402

# test_videos/ lives in the data tree. / 视频目录在数据树中。
ROOT = accident_root()
VIDEO_ROOT = ROOT / "test_videos"
DEFAULT_OFFSETS_SEC = [-2.0, -1.0, 0.0, 1.0, 2.0]
# Dense window around collision onset (seconds)
ONSET_OFFSETS_DENSE = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]
MAX_PIXELS = 512 * 512


def resolve_video_path(path_str: str) -> Path | None:
    p = Path(path_str)
    candidates = [
        VIDEO_ROOT / path_str,
        VIDEO_ROOT / p.name,
        VIDEO_ROOT / "videos" / p.name,
        ROOT / path_str,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def extract_frames_at_offsets(
    video_path: Path,
    onset_sec: float,
    offsets: list[float] | None = None,
    fps: float | None = None,
) -> tuple[list[Image.Image], list[float], float]:
    offsets = offsets or DEFAULT_OFFSETS_SEC
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], 0.0

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / vid_fps if vid_fps > 0 else 0.0

    frames: list[Image.Image] = []
    times: list[float] = []
    for off in offsets:
        t = max(0.0, min(onset_sec + off, max(0.0, duration - 0.05)))
        fidx = min(max(0, int(t * vid_fps)), max(0, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        w, h = pil.size
        if w * h > MAX_PIXELS:
            scale = (MAX_PIXELS / (w * h)) ** 0.5
            pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        frames.append(pil)
        times.append(round(t, 2))
    cap.release()
    return frames, times, duration


def extract_uniform_frames(
    video_path: Path,
    num_frames: int = 4,
    max_pixels: int = MAX_PIXELS,
) -> tuple[list[Image.Image], list[float], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], [], 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / fps if fps > 0 else 0.0
    indices = np.linspace(0, max(0, total - 1), num_frames, dtype=int)

    frames: list[Image.Image] = []
    times: list[float] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        w, h = pil.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            pil = pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        frames.append(pil)
        times.append(round(idx / fps, 2))
    cap.release()
    return frames, times, duration
