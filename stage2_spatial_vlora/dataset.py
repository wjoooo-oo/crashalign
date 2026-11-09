#!/usr/bin/env python3
"""
0917_Dataset for spatial VLoRA ablation (GT onset + integer answer).

Fixed prompt; assistant target is "x y" on a 0–1000 integer grid.
固定短 prompt；答案为 0–1000 整数网格上的 "x y"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

COORD_SCALE = 1000

# Minimal fixed prompt (user request) + one format line for stable parsing.
# 用户指定的固定短 prompt，外加一行格式约束便于解析。
SPATIAL_INSTRUCTION = (
    "Locate the collision point in the image.\n"
    "Reply with two integers in [0, 1000] as: x y\n"
    "where (0 0) is top-left and (1000 1000) is bottom-right."
)


class SpatialIntDataset(Dataset):
    """jsonl with frames=[gt_onset.jpg] and integer answer string."""

    def __init__(self, jsonl_path: str | Path):
        self.records: list[dict[str, Any]] = []
        with Path(jsonl_path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        frames = [Image.open(p).convert("RGB") for p in rec["frames"]]
        answer = rec["target"]["answer"]
        return {
            "frames": frames,
            "answer": answer,
            "path": rec["path"],
            "center_x": float(rec["target"]["center_x"]),
            "center_y": float(rec["target"]["center_y"]),
        }


class SpatialCollator:
    """Mask prompt tokens (−100); CE only on answer tokens. / prompt 置 −100，仅答案计 CE。"""

    def __init__(self, processor, instruction: str = SPATIAL_INSTRUCTION):
        self.processor = processor
        self.instruction = instruction
        self.pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    def _encode_one(self, frames: list[Image.Image], answer: str) -> dict[str, torch.Tensor]:
        user_content = [{"type": "image", "image": img} for img in frames]
        user_content.append({"type": "text", "text": self.instruction})
        user_msg = {"role": "user", "content": user_content}
        asst_msg = {"role": "assistant", "content": [{"type": "text", "text": answer}]}

        full = self.processor.apply_chat_template(
            [user_msg, asst_msg],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        prompt = self.processor.apply_chat_template(
            [user_msg],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        plen = prompt["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :plen] = -100
        full["labels"] = labels
        return full

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_one(b["frames"], b["answer"]) for b in batch]
        max_len = max(e["input_ids"].shape[1] for e in encoded)
        input_ids, attn, labels, mm_ids = [], [], [], []
        pixel_values, grid_thw = [], []
        has_mm = "mm_token_type_ids" in encoded[0]
        for e in encoded:
            ids = e["input_ids"][0]
            pad = max_len - ids.shape[0]
            input_ids.append(torch.cat([ids, torch.full((pad,), self.pad_id, dtype=ids.dtype)]))
            am = e.get("attention_mask")
            am = am[0] if am is not None else torch.ones_like(ids)
            attn.append(torch.cat([am, torch.zeros(pad, dtype=am.dtype)]))
            lb = e["labels"][0]
            labels.append(torch.cat([lb, torch.full((pad,), -100, dtype=lb.dtype)]))
            if has_mm:
                mt = e["mm_token_type_ids"][0]
                mm_ids.append(torch.cat([mt, torch.zeros(pad, dtype=mt.dtype)]))
            if "pixel_values" in e:
                pixel_values.append(e["pixel_values"])
                grid_thw.append(e["image_grid_thw"])
        out: dict[str, torch.Tensor] = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
        }
        if has_mm:
            out["mm_token_type_ids"] = torch.stack(mm_ids)
        if pixel_values:
            out["pixel_values"] = torch.cat(pixel_values, dim=0)
            out["image_grid_thw"] = torch.cat(grid_thw, dim=0)
        return out
