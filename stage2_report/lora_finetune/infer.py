#!/usr/bin/env python3
"""
0625_Inference + quick eval for a trained Qwen3.5 crash-report LoRA adapter.

Loads the 4-bit base + adapter, generates a JSON report per test video from frames,
and reports fixed-field accuracy (scene/weather/lighting/collision_type) vs the GT target.
加载 4-bit base + 适配器，对测试视频按帧生成 JSON 报告，并报告固定字段准确率。

Usage:
  python infer.py --adapter runs/stage2_gold --test_jsonl data/gold_test.jsonl \
      --out runs/stage2_gold/preds_gold_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # supportive_eval
from dataset import INSTRUCTION  # noqa: E402
from supportive_eval import extract_json_from_text  # noqa: E402
from train_lora import DEFAULT_MAX_PIXELS_FACTOR, configure_processor_pixels  # noqa: E402

MODEL_ID = os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-27B")
HF_CACHE = os.environ.get("HF_HOME", "/home/disk3/Jiachen/cache/huggingface")
FIXED_FIELDS = ["scene", "weather", "lighting", "collision_type"]


def load(adapter: str, max_pixels_factor: int = DEFAULT_MAX_PIXELS_FACTOR):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE)
    configure_processor_pixels(processor, max_pixels_factor=max_pixels_factor)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map={"": 0},
        trust_remote_code=True,
        cache_dir=HF_CACHE,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


@torch.no_grad()
def generate_one(model, processor, frames: list[Image.Image], max_new_tokens: int = 768) -> str:
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": INSTRUCTION})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    n = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode([out[0][n:]], skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(description="Run + eval a trained crash-report adapter")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_pixels_factor", type=int, default=DEFAULT_MAX_PIXELS_FACTOR)
    args = parser.parse_args()

    model, processor = load(args.adapter, max_pixels_factor=args.max_pixels_factor)
    records = [json.loads(l) for l in Path(args.test_jsonl).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        records = records[:args.limit]

    preds, hits = [], {f: 0 for f in FIXED_FIELDS}
    n_parse = 0
    for i, rec in enumerate(records):
        frames = [Image.open(p).convert("RGB") for p in rec["frames"]]
        raw = generate_one(model, processor, frames)
        parsed = extract_json_from_text(raw) or {}
        if parsed:
            n_parse += 1
        for f in FIXED_FIELDS:
            if parsed.get(f) is not None and str(parsed.get(f)) == str(rec["target"].get(f)):
                hits[f] += 1
        preds.append({"path": rec["path"], "pred": parsed, "target": rec["target"], "raw": raw})
        print(f"[{i+1}/{len(records)}] {rec['path']} parse={bool(parsed)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in preds) + "\n", encoding="utf-8")

    n = len(records)
    print("\n=== fixed-field accuracy ===", flush=True)
    print(f"parse_rate: {n_parse}/{n} = {n_parse/n:.3f}")
    for f in FIXED_FIELDS:
        print(f"{f}_acc: {hits[f]}/{n} = {hits[f]/n:.3f}")
    print(f"Predictions -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
