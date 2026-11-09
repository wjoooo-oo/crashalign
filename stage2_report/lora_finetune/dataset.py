#!/usr/bin/env python3
"""
0625_Dataset + collator for Qwen3.5 crash-report LoRA fine-tuning.

The student sees ONLY the frames (plus a fixed instruction) and must output the full
structured crash report as JSON. Fixed fields (scene/weather/lighting/collision_type)
are official GT targets the model learns to predict; it does not receive them as input.
student 仅看到关键帧（+固定指令），需输出完整结构化 JSON 报告；固定字段为官方 GT 目标，
不作为输入提供。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

# Reuse v7 participant-focus + mechanism guide from the 200-GT ablation winners.
# 复用 200-GT 消融中表现最好的 v7 参与者聚焦 + 冲突机理指南。
_EXP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_EXP_DIR))
from crash_description_prompts import COLLISION_MECHANISM_GUIDE  # noqa: E402
from prompts import PARTICIPANT_FOCUS  # noqa: E402

# Adapt v7 wording to our crash-report JSON schema (participants[] not vehicles[]).
# 将 v7 表述适配到 crash-report JSON schema（participants[] 而非 vehicles[]）。
PARTICIPANT_FOCUS_LORA = (
    PARTICIPANT_FOCUS.replace(
        "In vehicles[], list ONLY crash participants with their class (car/truck/bus/motorcycle) and role.",
        "In participants[], list ONLY crash participants using car/truck/bus/motorcycle/bicycle/pedestrian.",
    )
)

# Canonical output vocabularies (shown to the model to constrain the format).
# 标准输出词表（写入指令以约束输出格式）。
SCENE_OPTIONS = [
    "highway", "signalized intersection", "simple intersection",
    "grade-separated interchange", "city street", "tunnel", "parking lot",
]
WEATHER_OPTIONS = ["clear", "rain", "snow", "fog"]
LIGHTING_OPTIONS = ["day", "night", "dusk"]
COLLISION_OPTIONS = ["rear-end", "t-bone", "head-on", "sideswipe", "single-vehicle"]
PARTICIPANT_OPTIONS = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]

# Scene / class disambiguation (0712): targets the two failure modes vs TraffiX —
# highway↔grade-separated interchange, and pickup/SUV mislabeled as truck.
# 场景/车型消歧：针对相对 TraffiX 的两类主要错误。
SCENE_CLASS_GUIDE = (
    "SCENE DISAMBIGUATION (pick exactly one):\n"
    "  - highway: mainline freeway/expressway WITHOUT a visible interchange/ramp/overpass conflict.\n"
    "  - grade-separated interchange: crash involves a ramp, overpass, underpass, or merge from "
    "an interchange (NOT plain highway mainline).\n"
    "  - signalized intersection: junction WITH visible traffic lights / signal heads.\n"
    "  - simple intersection: junction WITHOUT traffic lights (stop/yield / unmarked).\n"
    "  - city street / tunnel / parking lot: as labeled.\n"
    "CLASS RULES for participants[]:\n"
    "  - pickup / SUV / van / sedan / coupe / hatchback / wagon / jeep → car (NEVER truck).\n"
    "  - truck ONLY for large commercial vehicles: semi, tractor-trailer, dump truck, tanker, "
    "box truck, flatbed, articulated lorry.\n"
    "  - rear-end / t-bone / head-on / sideswipe → list >=2 participants; single-vehicle → exactly 1.\n"
    "  - participants[] must be consistent with observation wording "
    "(if you write 'pickup'/'SUV'/'sedan', put car not truck).\n"
)

# v7-style instruction: participant focus + mechanism guide + 8-field JSON target.
# v7 风格指令：参与者聚焦 + 冲突机理指南 + 8 字段 JSON 目标。
INSTRUCTION = (
    "You are a traffic-accident analyst reviewing CCTV footage of a CONFIRMED collision.\n"
    "The frames below are fixed samples around the crash onset, ordered in time.\n\n"
    + PARTICIPANT_FOCUS_LORA
    + "\n"
    + COLLISION_MECHANISM_GUIDE
    + "\n"
    + SCENE_CLASS_GUIDE
    + "\nProduce a single structured crash report as JSON with EXACTLY these fields:\n"
    f"  scene: one of {SCENE_OPTIONS}\n"
    f"  weather: one of {WEATHER_OPTIONS}\n"
    f"  lighting: one of {LIGHTING_OPTIONS}\n"
    f"  participants: list using only {PARTICIPANT_OPTIONS}\n"
    f"  collision_type: one of {COLLISION_OPTIONS}\n"
    "  observation: 1-3 short sentences on each participant's path and the contact point "
    "(keep concrete motion language; do not collapse into keywords only)\n"
    "  explanation: one sentence naming the core conflict mechanism\n"
    "  summary: one sentence overview\n"
    "Reply with ONLY valid JSON."
)


def target_to_json(target: dict[str, Any]) -> str:
    """Serialize the target report to compact, stable JSON (assistant answer)."""
    # 将目标报告序列化为稳定的紧凑 JSON（作为 assistant 答案）
    ordered = {
        "scene": target.get("scene"),
        "weather": target.get("weather"),
        "lighting": target.get("lighting"),
        "participants": target.get("participants") or [],
        "collision_type": target.get("collision_type"),
        "observation": target.get("observation") or [],
        "explanation": target.get("explanation") or "",
        "summary": target.get("summary") or "",
    }
    return json.dumps(ordered, ensure_ascii=False, indent=2)


class CrashReportDataset(Dataset):
    """Reads a prepared jsonl (path, frames, target) into training samples."""

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
        answer = target_to_json(rec["target"])
        return {"frames": frames, "answer": answer, "path": rec["path"]}


class CrashCollator:
    """
    Build Qwen chat messages (frames + instruction -> JSON answer) and tokenize,
    masking the prompt tokens in labels so loss is computed only on the answer.
    构造 Qwen 对话（帧+指令 -> JSON 答案）并 tokenize，labels 中屏蔽 prompt 部分，
    仅在答案上计算 loss。
    """

    def __init__(self, processor):
        self.processor = processor
        self.pad_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id

    def _encode_one(self, frames: list[Image.Image], answer: str) -> dict[str, torch.Tensor]:
        user_content = [{"type": "image", "image": img} for img in frames]
        user_content.append({"type": "text", "text": INSTRUCTION})
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
                # mm_token_type_ids: 0=text token; pad with 0 so M-RoPE treats padding as text.
                # mm_token_type_ids：0=文本 token；padding 用 0，使 M-RoPE 将其视为文本。
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
