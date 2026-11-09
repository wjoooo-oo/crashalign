#!/usr/bin/env python3
"""
0914_Infer + evaluate spatial QLoRA vs zero-shot on the same onset frames.

Loads frozen Qwen3.5-27B (± spatial adapter), runs single-frame grounding, reports
isotropic S (σ=0.1) against GT. Does NOT load the crash-understanding LoRA.
加载冻结的 Qwen3.5-27B（± spatial adapter），在相同 onset 帧上评测各向同性 S；
不加载事故理解 LoRA。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

HERE = Path(__file__).resolve().parent
from dataset import SPATIAL_INSTRUCTION  # noqa: E402
from train_lora import (  # noqa: E402
    DEFAULT_MAX_PIXELS_FACTOR,
    MODEL_ID,
    HF_CACHE,
    configure_processor_pixels,
)

DATA_DIR = HERE / "data"


def spatial_score(x_gt: float, y_gt: float, x_pred: float, y_pred: float, sigma: float = 0.1) -> float:
    """Isotropic ACCIDENT S. / 各向同性空间分。"""
    dx = float(x_gt) - float(x_pred)
    dy = float(y_gt) - float(y_pred)
    return math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))


def parse_spatial_response(text: str) -> dict[str, float]:
    """Parse model JSON / fallback regex for (x,y,conf). / 解析坐标 JSON。"""
    clean = text
    if "</think>" in clean:
        clean = clean.split("</think>")[-1].strip()
    m = re.search(r'\{[^{}]*"center_x"[^{}]*\}', clean, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            return {
                "center_x": float(data.get("center_x") or 0.5),
                "center_y": float(data.get("center_y") or 0.5),
                "confidence": float(data.get("confidence") or 0.5),
            }
        except Exception:
            pass
    x_m = re.search(r'"center_x"\s*:\s*([\d.]+)', clean)
    y_m = re.search(r'"center_y"\s*:\s*([\d.]+)', clean)
    c_m = re.search(r'"confidence"\s*:\s*([\d.]+)', clean)
    return {
        "center_x": float(x_m.group(1)) if x_m else 0.5,
        "center_y": float(y_m.group(1)) if y_m else 0.5,
        "confidence": float(c_m.group(1)) if c_m else 0.0,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_model(adapter: str | None, max_pixels_factor: int):
    """Load 4-bit base; optionally attach spatial LoRA only. / 加载基座，可选挂载 spatial LoRA。"""
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
    if adapter:
        print(f"Loading spatial adapter: {adapter}", flush=True)
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, processor


@torch.no_grad()
def predict_one(model, processor, image_path: str) -> tuple[dict[str, float], str]:
    """Single-frame spatial inference (same input contract as qwen35_spatial)."""
    # 单帧空间推理（与 qwen35_spatial 输入约定一致）
    img = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": SPATIAL_INSTRUCTION},
        ],
    }]
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
    out_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    raw = processor.batch_decode(
        [out_ids[0][input_len:]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    pred = parse_spatial_response(raw)
    pred["center_x"] = max(0.01, min(0.99, pred["center_x"]))
    pred["center_y"] = max(0.01, min(0.99, pred["center_y"]))
    return pred, raw


def evaluate(model, processor, records: list[dict[str, Any]], tag: str, out_jsonl: Path) -> dict[str, Any]:
    """Run all samples and aggregate S / distance stats. / 全量推理并汇总 S 与距离。"""
    rows = []
    scores, dists = [], []
    t0 = time.time()
    for i, rec in enumerate(records):
        gt = rec["target"]
        pred, raw = predict_one(model, processor, rec["frames"][0])
        s = spatial_score(gt["center_x"], gt["center_y"], pred["center_x"], pred["center_y"])
        d = math.sqrt(
            (gt["center_x"] - pred["center_x"]) ** 2
            + (gt["center_y"] - pred["center_y"]) ** 2
        )
        scores.append(s)
        dists.append(d)
        rows.append({
            "path": rec["path"],
            "gt_x": gt["center_x"],
            "gt_y": gt["center_y"],
            "pred_x": pred["center_x"],
            "pred_y": pred["center_y"],
            "confidence": pred.get("confidence", 0.0),
            "S": round(s, 6),
            "dist": round(d, 6),
            "raw": raw,
        })
        eta = (time.time() - t0) / (i + 1) * (len(records) - i - 1) / 60
        print(
            f"[{tag} {i+1}/{len(records)}] "
            f"pred=({pred['center_x']:.3f},{pred['center_y']:.3f}) "
            f"gt=({gt['center_x']:.3f},{gt['center_y']:.3f}) "
            f"S={s:.3f} d={d:.3f} ETA={eta:.1f}m",
            flush=True,
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dists_sorted = sorted(dists)
    summary = {
        "tag": tag,
        "n": len(records),
        "S_mean": float(sum(scores) / max(len(scores), 1)),
        "dist_mean": float(sum(dists) / max(len(dists), 1)),
        "dist_median": float(dists_sorted[len(dists_sorted) // 2]) if dists_sorted else 0.0,
        "frac_d_lt_0.05": float(sum(1 for d in dists if d < 0.05) / max(len(dists), 1)),
        "frac_d_lt_0.10": float(sum(1 for d in dists if d < 0.10) / max(len(dists), 1)),
        "frac_d_lt_0.20": float(sum(1 for d in dists if d < 0.20) / max(len(dists), 1)),
        "sigma": 0.1,
        "preds": str(out_jsonl),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial ZS vs LoRA eval on onset frames")
    parser.add_argument("--adapter", type=str, default=None, help="Spatial LoRA dir; omit for ZS-only")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--compare_zeroshot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run zero-shot. Default: True if --adapter is omitted, else False.",
    )
    parser.add_argument("--max_pixels_factor", type=int, default=DEFAULT_MAX_PIXELS_FACTOR)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit")
    parser.add_argument("--out_dir", type=str, default=str(HERE / "runs" / "eval"))
    args = parser.parse_args()

    # Default: ZS when no adapter; LoRA-only when adapter is given (avoids dual-load OOM).
    # 默认：无 adapter 时跑 ZS；有 adapter 时只跑 LoRA（避免同进程双载 OOM）。
    if args.compare_zeroshot is None:
        args.compare_zeroshot = args.adapter is None
    if not args.compare_zeroshot and not args.adapter:
        raise SystemExit("Nothing to run: provide --adapter and/or --compare-zeroshot")

    jsonl = DATA_DIR / f"{args.split}.jsonl"
    if not jsonl.is_file():
        raise FileNotFoundError(f"Missing {jsonl}; run prepare_data.py first")
    records = load_jsonl(jsonl)
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    out_dir = Path(args.out_dir)
    summaries = []

    if args.compare_zeroshot:
        print("=== Zero-shot (no adapter) ===", flush=True)
        zs_model, zs_proc = load_model(None, args.max_pixels_factor)
        zs_sum = evaluate(
            zs_model, zs_proc, records, "zeroshot",
            out_dir / f"{args.split}_zeroshot_preds.jsonl",
        )
        summaries.append(zs_sum)
        print(json.dumps(zs_sum, indent=2), flush=True)
        # Fully release ZS weights before loading LoRA in the same process.
        # 同进程加载 LoRA 前彻底释放零样本权重，避免 OOM。
        del zs_model, zs_proc
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if args.adapter:
        print(f"=== Spatial LoRA: {args.adapter} ===", flush=True)
        lora_model, lora_proc = load_model(args.adapter, args.max_pixels_factor)
        lora_sum = evaluate(
            lora_model, lora_proc, records, "spatial_lora",
            out_dir / f"{args.split}_lora_preds.jsonl",
        )
        summaries.append(lora_sum)
        print(json.dumps(lora_sum, indent=2), flush=True)
        del lora_model, lora_proc
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    report = {"split": args.split, "n": len(records), "methods": summaries}
    if len(summaries) == 2:
        report["delta_S"] = summaries[1]["S_mean"] - summaries[0]["S_mean"]
        report["delta_dist"] = summaries[1]["dist_mean"] - summaries[0]["dist_mean"]
    report_path = out_dir / f"{args.split}_compare.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report -> {report_path}", flush=True)
    if "delta_S" in report:
        print(f"delta_S (LoRA - ZS) = {report['delta_S']:+.4f}", flush=True)


if __name__ == "__main__":
    main()
