#!/usr/bin/env python3
"""
0917_Infer + evaluate spatial VLoRA adapters on GT-onset frames.

Reports isotropic S (σ=0.1) for zero-shot and/or a trained adapter.
在标注 onset 帧上评测零样本与/或训练好的 adapter。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image

HERE = Path(__file__).resolve().parent
from dataset import COORD_SCALE, SPATIAL_INSTRUCTION  # noqa: E402
from model_utils import DEFAULT_MAX_PIXELS_FACTOR, load_model_and_processor  # noqa: E402
from train import parse_xy, spatial_score  # noqa: E402

DATA_DIR = HERE / "data"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_for_infer(adapter: str | None, max_pixels_factor: int):
    model, processor = load_model_and_processor(max_pixels_factor=max_pixels_factor)
    if adapter:
        print(f"load adapter {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


@torch.no_grad()
def predict_one(model, processor, image_path: str) -> tuple[dict[str, float], str]:
    img = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": SPATIAL_INSTRUCTION},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    )
    inputs = inputs.to(model.device)
    in_len = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    raw = processor.batch_decode([out[0][in_len:]], skip_special_tokens=True)[0].strip()
    xy = parse_xy(raw) or (0.5, 0.5)
    return {"center_x": xy[0], "center_y": xy[1]}, raw


def evaluate(model, processor, records, tag: str, out_jsonl: Path) -> dict[str, Any]:
    rows, scores, dists = [], [], []
    t0 = time.time()
    for i, rec in enumerate(records):
        pred, raw = predict_one(model, processor, rec["frames"][0])
        gt_x, gt_y = rec["target"]["center_x"], rec["target"]["center_y"]
        s = spatial_score(gt_x, gt_y, pred["center_x"], pred["center_y"])
        d = math.hypot(gt_x - pred["center_x"], gt_y - pred["center_y"])
        scores.append(s)
        dists.append(d)
        rows.append({
            "path": rec["path"], "gt_x": gt_x, "gt_y": gt_y,
            "pred_x": pred["center_x"], "pred_y": pred["center_y"],
            "S": round(s, 6), "dist": round(d, 6), "raw": raw,
        })
        eta = (time.time() - t0) / (i + 1) * (len(records) - i - 1) / 60
        print(
            f"[{tag} {i+1}/{len(records)}] "
            f"pred=({pred['center_x']:.3f},{pred['center_y']:.3f}) "
            f"S={s:.3f} raw={raw!r} ETA={eta:.1f}m",
            flush=True,
        )
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dists_sorted = sorted(dists)
    return {
        "tag": tag,
        "n": len(records),
        "S_mean": float(sum(scores) / max(len(scores), 1)),
        "dist_mean": float(sum(dists) / max(len(dists), 1)),
        "dist_median": float(dists_sorted[len(dists_sorted) // 2]) if dists_sorted else 0.0,
        "frac_d_lt_0.10": float(sum(d < 0.10 for d in dists) / max(len(dists), 1)),
        "sigma": 0.1,
        "preds": str(out_jsonl),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out_dir", default=str(HERE / "runs" / "eval"))
    ap.add_argument("--max_pixels_factor", type=int, default=DEFAULT_MAX_PIXELS_FACTOR)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = load_jsonl(DATA_DIR / f"{args.split}.jsonl")
    if args.limit > 0:
        records = records[: args.limit]
    tag = "zeroshot" if not args.adapter else Path(args.adapter).name
    model, processor = load_for_infer(args.adapter, args.max_pixels_factor)
    summary = evaluate(model, processor, records, tag, Path(args.out_dir) / f"{args.split}_{tag}_preds.jsonl")
    out = Path(args.out_dir) / f"{args.split}_{tag}_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
