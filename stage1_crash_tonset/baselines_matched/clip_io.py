#!/usr/bin/env python3
"""Deterministic ACCIDENT matched clip catalog and frame I/O.

Each source video yields one positive clip (covers onset) and one same-video
clean-normal negative (ends >= 3 s before onset). Baselines must read only
this time range — never the full video — or the negative would leak the crash.

每个源视频一条正样本（覆盖 onset）和一条同视频 clean-normal 负样本。
基线只能读该时间窗，不能读整段视频，否则负样本会泄漏事故画面。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # stage1 code dir: labels.py and the matched sampler live here
sys.path.insert(0, str(ROOT.parent))
from crashalign_paths import accident_root  # noqa: E402

# Videos live in the data tree. ROOT stays the stage-1 code directory.
# 视频在数据目录。ROOT 仍是 Stage I 代码目录（labels 与采样脚本在这里）。
ACCIDENT_ROOT = accident_root()
# Another matched split (e.g. the 0926 gap-2s benchmark) is selected via environment
# variables, so every baseline runner switches split without new flags.
# 通过环境变量切换到其它 matched 划分（如 0926 gap-2s benchmark），各基线脚本无需新增参数。
def _prefer_existing(env_name: str, packaged: Path, original: Path) -> Path:
    """Use an explicit path, else a directory that already exists.

    优先环境变量；否则用已经存在的目录（代码包内或原始 accident 树）。
    """
    env = os.environ.get(env_name, "").strip()
    if env:
        return Path(env)
    if packaged.exists():
        return packaged
    if original.exists():
        return original
    return packaged


_ORIG = accident_root() / "0723_task1_cumphase"
DATA_DIR = _prefer_existing(
    "MATCHED_DATA_DIR",
    ROOT / "data_accident_matched",
    _ORIG / "data_accident_matched",
)
CACHE_DIR = _prefer_existing(
    "MATCHED_CACHE_DIR",
    ROOT / "cache_emb_accident",
    _ORIG / "cache_emb_accident",
)
CATALOG_DIR = DATA_DIR / "clip_catalog"

sys.path.insert(0, str(ROOT))
from labels import CLEAN_END_MARGIN  # noqa: E402

import importlib.util

_spec = importlib.util.spec_from_file_location("sampling_matched", ROOT / "0726_sampling_matched.py")
_samp = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_samp)

FIXED_LEN = 8
WINDOW_SEC = 2.0
SEED = 42


@dataclass
class ClipSpec:
    clip_id: str
    video_id: str
    source_video_id: str
    path: str
    resolved_path: str
    split: str
    role: str
    is_crash: int
    onset: float
    t_start: float
    t_end: float
    start_index: int
    fps: float
    start_frame: int
    end_frame: int
    n_frames: int
    pos_neg_gap_sec: float | None = None  # gap mode (0925/0926 splits) / 间隔模式

    @property
    def video_path(self) -> Path:
        return Path(self.resolved_path)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def resolve_video(rel_path: str) -> Path:
    candidates = [
        ACCIDENT_ROOT / rel_path,
        ACCIDENT_ROOT / "test_videos" / "videos" / Path(rel_path).name,
        ACCIDENT_ROOT / Path(rel_path).name,
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"video not found: {rel_path}")


def clip_time_range(times: np.ndarray, start: int, L: int, window_sec: float = WINDOW_SEC) -> tuple[float, float]:
    half = window_sec / 2.0
    t0 = float(times[start]) - half
    t1 = float(times[start + L - 1]) + half
    return max(0.0, t0), t1


def _video_meta(path: Path) -> tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 1e-3:
        fps = 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, n


def build_split_catalog(split: str, *, train_mode: bool = False) -> list[ClipSpec]:
    """Materialize deterministic (val/test) or seeded-train clip specs.

    生成 clip 目录：val/test 用 hash 稳定选取；train 默认也用稳定选取以便复现。
    """
    rows = load_jsonl(DATA_DIR / f"{split}_matched.jsonl")
    cache_split = CACHE_DIR / split
    rng = np.random.default_rng(SEED)
    specs: list[ClipSpec] = []
    for row in rows:
        vid = row["video_id"]
        npz = cache_split / f"{vid}.npz"
        if not npz.is_file():
            raise FileNotFoundError(f"missing cache {npz}")
        with np.load(npz, allow_pickle=True) as z:
            times = z["times"].astype(np.float32)
            onset = float(z["t_gt"])
            if onset < 0:
                onset = float(row["accident_time"])
        vpath = resolve_video(row["path"])
        fps, n_vid = _video_meta(vpath)
        gap_sec = row.get("pos_neg_gap_sec")
        for role in ("positive", "negative"):
            clip = _samp.sample_matched_clip(
                np.zeros((len(times), 1), dtype=np.float32),
                times,
                onset=onset,
                role=role,
                L=FIXED_LEN,
                train=train_mode,
                rng=rng,
                video_id=vid,
                window_sec=WINDOW_SEC,
                gap_sec=gap_sec,
            )
            t0, t1 = clip_time_range(times, int(clip["start_index"]), FIXED_LEN)
            start_f = int(max(0, round(t0 * fps)))
            end_f = int(min(n_vid, max(start_f + 1, round(t1 * fps))))
            specs.append(
                ClipSpec(
                    clip_id=f"{vid}_{role}",
                    video_id=vid,
                    source_video_id=row["source_video_id"],
                    path=row["path"],
                    resolved_path=str(vpath),
                    split=split,
                    role=role,
                    is_crash=1 if role == "positive" else 0,
                    onset=float(onset),
                    t_start=float(t0),
                    t_end=float(t1),
                    start_index=int(clip["start_index"]),
                    fps=float(fps),
                    start_frame=start_f,
                    end_frame=end_f,
                    n_frames=max(0, end_f - start_f),
                    pos_neg_gap_sec=gap_sec,
                )
            )
    return specs


def save_catalog(split: str, specs: list[ClipSpec]) -> Path:
    out = CATALOG_DIR / f"{split}.jsonl"
    write_jsonl(out, [asdict(s) for s in specs])
    return out


def load_catalog(split: str) -> list[ClipSpec]:
    path = CATALOG_DIR / f"{split}.jsonl"
    if not path.is_file():
        specs = build_split_catalog(split)
        save_catalog(split, specs)
        return specs
    return [ClipSpec(**r) for r in load_jsonl(path)]


def ensure_catalogs() -> dict[str, list[ClipSpec]]:
    out = {}
    for split in ("train", "val", "test"):
        out[split] = load_catalog(split)
    return out


def read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sample_indices(start: int, end: int, n: int) -> list[int]:
    if end <= start:
        return [max(0, start)]
    if n <= 1:
        return [start]
    idx = np.linspace(start, end - 1, n, dtype=int)
    return sorted(set(int(i) for i in idx.tolist()))


def load_frames_sequential(spec: ClipSpec, indices: list[int]) -> list[np.ndarray]:
    """Decode a clip range once and keep only requested frame indices.

    Sequential decode is much faster than cap.set() per frame.
    顺序解码远快于逐帧 seek。
    """
    if not indices:
        return []
    wanted = set(int(i) for i in indices)
    lo, hi = min(wanted), max(wanted)
    cap = cv2.VideoCapture(spec.resolved_path)
    if not cap.isOpened():
        return []
    cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
    found: dict[int, np.ndarray] = {}
    idx = lo
    while idx <= hi:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            found[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
    cap.release()
    return [found[i] for i in indices if i in found]


def load_clip_frames_ffmpeg(
    spec: ClipSpec,
    n_frames: int,
    width: int = 128,
    height: int = 128,
) -> list[np.ndarray]:
    """Fast keyframe seek + decode only the matched time window via ffmpeg.

    用 ffmpeg -ss 关键帧定位，只解码 matched 时间窗。
    """
    dur = max(0.08, spec.t_end - spec.t_start)
    fps = n_frames / dur
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{spec.t_start:.3f}",
        "-i",
        spec.resolved_path,
        "-t",
        f"{dur:.3f}",
        "-vf",
        f"fps={fps:.4f},scale={width}:{height}",
        "-frames:v",
        str(n_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    raw = proc.stdout
    nbytes = width * height * 3
    frames = []
    for i in range(0, len(raw) - nbytes + 1, nbytes):
        frames.append(np.frombuffer(raw[i : i + nbytes], dtype=np.uint8).reshape(height, width, 3).copy())
    if frames:
        return frames
    return load_frames_sequential(spec, sample_indices(spec.start_frame, spec.end_frame, n_frames))


def load_clip_frames_rgb(spec: ClipSpec, n_frames: int) -> list[np.ndarray]:
    """Uniformly sample n RGB frames from the matched clip only."""
    return load_clip_frames_ffmpeg(spec, n_frames, width=256, height=256)


def load_clip_frames_pil(spec: ClipSpec, n_frames: int, max_side: int = 512) -> list[Image.Image]:
    frames = []
    for rgb in load_clip_frames_rgb(spec, n_frames):
        im = Image.fromarray(rgb)
        w, h = im.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
        frames.append(im)
    return frames


def iter_clip_frame_indices(spec: ClipSpec, stride: int = 1) -> list[int]:
    return list(range(spec.start_frame, spec.end_frame, max(1, stride)))


def validate_specs(specs: list[ClipSpec]) -> list[str]:
    """Return human-readable protocol violations, if any."""
    errors = []
    by_vid: dict[str, dict[str, ClipSpec]] = {}
    for s in specs:
        by_vid.setdefault(s.video_id, {})[s.role] = s
        if s.n_frames < 2:
            errors.append(f"{s.clip_id}: too few frames ({s.n_frames})")
        last_center = s.t_end - WINDOW_SEC / 2.0
        if s.role == "positive" and not (s.t_start - 1e-3 <= s.onset <= s.t_end + 1e-3):
            errors.append(f"{s.clip_id}: positive does not cover onset {s.onset:.3f} in [{s.t_start:.3f},{s.t_end:.3f}]")
        if s.role == "negative" and last_center > s.onset - CLEAN_END_MARGIN + 1e-3:
            errors.append(
                f"{s.clip_id}: neg last-center {last_center:.3f} > onset-{CLEAN_END_MARGIN}={s.onset - CLEAN_END_MARGIN:.3f}"
            )
    for vid, roles in by_vid.items():
        if set(roles) != {"positive", "negative"}:
            errors.append(f"{vid}: missing role {set(roles)}")
            continue
        # Gap mode: the negative must end >= gap before the positive starts.
        # 间隔模式：负样本须在正样本开始前 ≥ gap 秒结束。
        pos, neg = roles["positive"], roles["negative"]
        gap = pos.pos_neg_gap_sec
        if gap is not None and neg.t_end > pos.t_start - gap + 1e-3:
            errors.append(f"{vid}: neg ends {neg.t_end:.3f} < {gap}s before pos start {pos.t_start:.3f}")
    return errors
