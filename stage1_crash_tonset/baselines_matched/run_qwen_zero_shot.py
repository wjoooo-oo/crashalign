#!/usr/bin/env python3
"""Zero-shot Qwen3.5-27B crash/non-crash on ACCIDENT matched clips.

Frames are taken only from the matched time window so the negative clip
cannot leak the later collision in the same source video.
只从 matched 时间窗取帧，避免同视频负样本泄漏事故画面。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from clip_io import load_catalog, load_clip_frames_pil  # noqa: E402
from metrics_matched import save_run  # noqa: E402

ROOT = HERE.parent
OUT_DEFAULT = ROOT / "results/accident_matched_baselines/qwen35_27b_zeroshot"
MODEL_ID = os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-27B")
HF_CACHE = os.environ.get("HF_HOME", "/home/disk3/Jiachen/cache/huggingface")

PROMPT = """You are analyzing CCTV traffic footage sampled as multiple frames from one short clip.

Task: Does this clip contain a traffic accident (collision involving vehicles)?

Reply with EXACTLY two lines:
Line 1: yes or no
Line 2: confidence: a number from 0.0 to 1.0

Example:
yes
confidence: 0.82
"""


def load_model():
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading {MODEL_ID} (4-bit) ...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, cache_dir=HF_CACHE)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=quant,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=HF_CACHE,
    ).eval()
    return model, processor


def run_vlm(model, processor, frames, prompt: str) -> str:
    content = [{"type": "image", "image": img} for img in frames]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = inputs.to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, temperature=0.0, do_sample=False)
    return processor.batch_decode(
        [out[0][input_len:]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_response(text: str) -> tuple[int, float]:
    lines = [ln.strip().lower() for ln in text.splitlines() if ln.strip()]
    yes = 0
    conf = 0.5
    if lines:
        if lines[0].startswith("yes"):
            yes = 1
        elif lines[0].startswith("no"):
            yes = 0
    m = re.search(r"confidence:\s*([0-9]*\.?[0-9]+)", text, re.I)
    if m:
        conf = float(m.group(1))
    score = conf if yes else (1.0 - conf)
    return yes, max(0.0, min(1.0, score))


def score_specs(model, processor, specs, n_frames: int) -> list[dict]:
    rows = []
    t0 = time.time()
    for i, s in enumerate(tqdm(specs, desc="qwen clip")):
        frames = load_clip_frames_pil(s, n_frames)
        if not frames:
            raw, score = "", 0.0
        else:
            raw = run_vlm(model, processor, frames, PROMPT)
            _, score = parse_response(raw)
        rows.append(
            {
                "clip_id": s.clip_id,
                "video_id": s.video_id,
                "role": s.role,
                "is_crash": int(s.is_crash),
                "score": float(score),
                "raw": raw,
            }
        )
        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(specs) - i - 1)
            print(f"  {i+1}/{len(specs)} elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--output_dir", type=Path, default=OUT_DEFAULT)
    args = p.parse_args()

    val, test = load_catalog("val"), load_catalog("test")
    print(f"Qwen zero-shot val={len(val)} test={len(test)} (no training)", flush=True)
    model, processor = load_model()
    val_rows = score_specs(model, processor, val, args.num_frames)
    test_rows = score_specs(model, processor, test, args.num_frames)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw_responses.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val_rows + test_rows)
    )
    save_run(
        out,
        method="Qwen3.5-27B zero-shot",
        val_rows=val_rows,
        test_rows=test_rows,
        extra={"num_frames": args.num_frames, "model": MODEL_ID, "setting": "pretrained"},
    )


if __name__ == "__main__":
    main()
