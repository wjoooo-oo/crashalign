#!/usr/bin/env python3
"""PyTorch dataset for ACCIDENT matched fixed-L pos/neg clips.

Each matched source video contributes two items:
  - positive: L windows covering onset
  - negative: L-window clean-normal ending >= 3 s before onset

No padding; valid_mask is all-True. Phase labels recomputed from onset.
每个 matched 源视频提供一条正样本与一条同视频 clean-normal 负样本。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _load_sampling():
    path = HERE / "0726_sampling_matched.py"
    spec = importlib.util.spec_from_file_location("sampling_matched", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_samp = _load_sampling()
sample_matched_clip = _samp.sample_matched_clip


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


class MatchedAccidentDataset(Dataset):
    """Expand each matched video into (positive, negative) training items."""

    def __init__(
        self,
        matched_jsonl: Path,
        cache_split_dir: Path,
        fixed_len: int = 8,
        train: bool = True,
        seed: int = 42,
        window_sec: float = 2.0,
    ):
        self.rows = load_jsonl(Path(matched_jsonl))
        self.cache_dir = Path(cache_split_dir)
        self.fixed_len = fixed_len
        self.train = train
        self.window_sec = window_sec
        self.rng = np.random.default_rng(seed)
        # Flatten to (video_row_index, role) / 展平为 (视频行, 角色)
        self.items: list[tuple[int, str]] = []
        for i, _ in enumerate(self.rows):
            self.items.append((i, "positive"))
            self.items.append((i, "negative"))
        if not self.items:
            raise FileNotFoundError(f"No matched rows in {matched_jsonl}")

    def __len__(self) -> int:
        return len(self.items)

    def labels(self) -> list[int]:
        return [1 if role == "positive" else 0 for _, role in self.items]

    def __getitem__(self, idx: int):
        row_i, role = self.items[idx]
        row = self.rows[row_i]
        vid = row["video_id"]
        npz = self.cache_dir / f"{vid}.npz"
        with np.load(npz, allow_pickle=True) as z:
            emb = z["emb"].astype(np.float32)
            times = z["times"].astype(np.float32)
            onset = float(z["t_gt"])
            if onset < 0:
                onset = float(row["accident_time"])

        clip = sample_matched_clip(
            emb,
            times,
            onset=onset,
            role=role,
            L=self.fixed_len,
            train=self.train,
            rng=self.rng,
            video_id=vid,
            window_sec=self.window_sec,
            # Rows built by 0925_build_gap_split.py carry a pos/neg gap; older rows do not.
            # 0925 重划分的行带有正负间隔字段；旧数据没有该字段，保持原采样。
            gap_sec=row.get("pos_neg_gap_sec"),
        )
        return {
            "emb": torch.from_numpy(clip["emb"]),
            "y_a": torch.from_numpy(clip["y_a"]),
            "y_b": torch.from_numpy(clip["y_b"]),
            "y_phase": torch.from_numpy(clip["y_phase"]),
            "phase_valid": torch.from_numpy(clip["phase_valid"]),
            "valid_mask": torch.from_numpy(clip["valid_mask"]),
            "is_crash": torch.tensor(float(clip["is_crash"]), dtype=torch.float32),
            "length": self.fixed_len,
            "video_id": f"{vid}_{role}",
            "source_video_id": vid,
            "role": role,
            "start_index": clip["start_index"],
        }
