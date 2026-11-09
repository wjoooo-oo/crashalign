#!/usr/bin/env python3
"""
0625_Two-stage QLoRA fine-tuning for Qwen3.5 crash-report student.

One generic trainer used for both stages / 同一脚本用于两阶段:
  Stage 1 (silver): --train_jsonl silver_train.jsonl  (learn format + recognition)
  Stage 2 (gold):   --train_jsonl gold_train.jsonl --init_adapter <stage1>  --lr small
                    (correct style / hallucination with human GT, smaller LR)

Memory strategy (single 49GB A6000) / 单卡显存策略:
  4-bit QLoRA + frozen vision + gradient checkpointing + 20 frames
  + max_pixels_factor=160 (caps vision tokens to avoid OOM).

Usage:
  python train_lora.py --train_jsonl data/silver_train.jsonl --output_dir runs/stage1_silver \
      --lr 1e-4 --epochs 1
  python train_lora.py --train_jsonl data/gold_train.jsonl  --val_jsonl data/gold_val.jsonl \
      --init_adapter runs/stage1_silver --output_dir runs/stage2_gold --lr 2e-5 --epochs 3
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Qwen3_5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

HERE = Path(__file__).resolve().parent
from dataset import CrashCollator, CrashReportDataset  # noqa: E402

MODEL_ID = os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-27B")
HF_CACHE = os.environ.get("HF_HOME", "/home/disk3/Jiachen/cache/huggingface")

# LoRA targets the language-model projections only; the vision tower uses different
# module names (qkv/proj/fc1/fc2) so it stays untouched and is additionally frozen.
# LoRA 仅作用于语言模型投影层；视觉塔用不同模块名，故不受影响，并额外冻结。
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Cap per-image resolution so 20 frames fit on a single 49GB GPU (smoke-tested).
# 限制单图分辨率，使 20 帧在单卡 49GB 上可训练（已通过冒烟测试）。
DEFAULT_MAX_PIXELS_FACTOR = 128  # smoke-tested OK for 20 frames on 49GB A6000
DEFAULT_MIN_PIXELS_FACTOR = 128


def configure_processor_pixels(
    processor,
    max_pixels_factor: int = DEFAULT_MAX_PIXELS_FACTOR,
    min_pixels_factor: int = DEFAULT_MIN_PIXELS_FACTOR,
) -> None:
    """
    Shrink vision tokens for multi-frame QLoRA on limited VRAM.
    压缩视觉 token，避免多帧 QLoRA 在有限显存上 OOM。
    """
    ip = processor.image_processor
    min_px = min_pixels_factor * 28 * 28
    max_px = max_pixels_factor * 28 * 28
    # Qwen3VL uses min_pixels/max_pixels on the fast image processor.
    # Qwen3VL 在 fast image processor 上使用 min_pixels/max_pixels。
    ip.min_pixels = min_px
    ip.max_pixels = max_px
    if hasattr(ip, "size"):
        try:
            ip.size["shortest_edge"] = min_px
            ip.size["longest_edge"] = max_px
        except Exception:
            pass
    print(f"processor pixels: min={min_px} max={max_px}", flush=True)


def load_model_and_processor(max_pixels_factor: int = DEFAULT_MAX_PIXELS_FACTOR):
    """Load Qwen3.5 in 4-bit and freeze the vision tower."""
    # 4-bit 加载 Qwen3.5 并冻结视觉塔
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
    # Freeze the vision tower (only the LM adapters train).
    # 冻结视觉塔（只训练语言模型的 LoRA）。
    if hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False
    return model, processor


def build_peft_model(model, init_adapter: str | None):
    """Attach LoRA adapters (fresh) or resume from a previous stage's adapter."""
    # 新建 LoRA 适配器，或从上一阶段的适配器继续
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if init_adapter:
        print(f"Resuming adapter from {init_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        lora = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        )
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


def main():
    parser = argparse.ArgumentParser(description="Two-stage QLoRA trainer for Qwen3.5 crash-report")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--init_adapter", type=str, default=None, help="Resume adapter (stage 2)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=-1, help="Override for smoke test")
    parser.add_argument(
        "--max_pixels_factor", type=int, default=DEFAULT_MAX_PIXELS_FACTOR,
        help="Vision max_pixels = factor * 28^2 (lower = less VRAM, default 160 for 20 frames)",
    )
    parser.add_argument(
        "--early_stopping_patience", type=int, default=None,
        help="If set, eval each epoch and stop when eval_loss stalls; keep best checkpoint",
    )
    args = parser.parse_args()

    use_early_stop = args.early_stopping_patience is not None and args.early_stopping_patience > 0

    model, processor = load_model_and_processor(max_pixels_factor=args.max_pixels_factor)
    model = build_peft_model(model, args.init_adapter)
    model.config.use_cache = False

    train_ds = CrashReportDataset(args.train_jsonl)
    val_ds = CrashReportDataset(args.val_jsonl) if args.val_jsonl else None
    collator = CrashCollator(processor)
    print(f"train={len(train_ds)} val={len(val_ds) if val_ds else 0}", flush=True)
    if use_early_stop and not val_ds:
        raise ValueError("--early_stopping_patience requires --val_jsonl")

    # Stage 2: eval every epoch on gold val, save best, early-stop on plateau.
    # 阶段2：每 epoch 在 gold val 上评估，保留 best，验证集 loss 停滞则早停。
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=args.logging_steps,
        save_steps=args.save_steps if not use_early_stop else 999999,
        save_strategy="epoch" if use_early_stop else "steps",
        save_total_limit=2,
        eval_strategy="epoch" if val_ds else "no",
        eval_on_start=use_early_stop,
        load_best_model_at_end=use_early_stop,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        per_device_eval_batch_size=1,
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        dataloader_num_workers=0,
    )

    callbacks = []
    if use_early_stop:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        print(
            f"early_stopping: patience={args.early_stopping_patience}, "
            f"eval=epoch, load_best_model_at_end=True",
            flush=True,
        )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    # With load_best_model_at_end, weights are already the best val checkpoint.
    # load_best_model_at_end=True 时，权重已是 val 最优 checkpoint。
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved adapter -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
