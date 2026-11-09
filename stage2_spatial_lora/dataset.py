#!/usr/bin/env python3
"""
0914_Dataset + collator for spatial-grounding QLoRA (single onset frame).

Student sees ONE onset frame + fixed spatial instruction and must emit
{"center_x","center_y","confidence"} JSON. Separate from crash-report LoRA.
学生仅见一张 onset 帧 + 固定空间指令，输出碰撞点 JSON；与事故报告 LoRA 分离。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

# Match the spirit of qwen35_spatial SPATIAL_PROMPT_V1 (generic, no GT leak).
# 对齐 qwen35_spatial 通用空间 prompt（不泄漏 GT / scene）。
SPATIAL_INSTRUCTION = (
    "Look at this traffic camera image showing a car accident scene.\n"
    "\n"
    "Find the exact point where the collision/impact happened or where the most "
    "damage/debris is visible.\n"
    "\n"
    "Give the collision point as normalized coordinates where:\n"
    "- (0,0) = top-left corner of the image\n"
    "- (1,1) = bottom-right corner of the image\n"
    "\n"
    "Look for: vehicles making contact, debris on road, crushed/deformed vehicle "
    "parts, skid marks converging.\n"
    "\n"
    "Reply with ONLY JSON:\n"
    '{"center_x": 0.0, "center_y": 0.0, "confidence": 0.0}'
)


def target_to_json(target: dict[str, Any]) -> str:
    """Serialize GT point to compact JSON answer. / 将 GT 点序列化为 JSON 答案。"""
    ordered = {
        "center_x": float(target["center_x"]),
        "center_y": float(target["center_y"]),
        "confidence": float(target.get("confidence", 0.9)),
    }
    return json.dumps(ordered, ensure_ascii=False)


class SpatialGroundingDataset(Dataset):
    """jsonl records with frames=[onset.jpg] and target coords."""

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
        # Original pipeline input: exactly one onset frame.
        # 原始管线输入：恰好一张 onset 帧。
        frames = [Image.open(p).convert("RGB") for p in rec["frames"]]
        answer = target_to_json(rec["target"])
        return {"frames": frames, "answer": answer, "path": rec["path"]}


class SpatialCollator:
    """
    Build chat (image + spatial instruction -> JSON) and mask prompt tokens in labels.
    构造对话并对 labels 中的 prompt token 置 -100，仅在答案上计 loss。
    """

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
            n = ids.shape[0]
            pad = max_len - n
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
