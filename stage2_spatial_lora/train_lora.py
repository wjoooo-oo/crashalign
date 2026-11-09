#!/usr/bin/env python3
"""
0914_QLoRA trainer for spatial grounding (separate adapter from crash understanding).

Frozen pretrained backbone + frozen vision encoder; only LM LoRA adapters train.
预训练骨干与视觉编码器冻结，仅训练语言模型 LoRA；与 crash-report adapter 完全分离。
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
from dataset import SpatialCollator, SpatialGroundingDataset  # noqa: E402

MODEL_ID = os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-27B")
HF_CACHE = os.environ.get("HF_HOME", "/home/disk3/Jiachen/cache/huggingface")

# LM-only LoRA targets; vision modules use different names and stay frozen.
# 仅语言模型 LoRA；视觉模块名不同，保持冻结。
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Single-frame → can afford higher resolution than 20-frame crash LoRA.
# 单帧输入，可比 crash LoRA 的 20 帧设置更高分辨率。
DEFAULT_MAX_PIXELS_FACTOR = 256
DEFAULT_MIN_PIXELS_FACTOR = 128


def configure_processor_pixels(
    processor,
    max_pixels_factor: int = DEFAULT_MAX_PIXELS_FACTOR,
    min_pixels_factor: int = DEFAULT_MIN_PIXELS_FACTOR,
) -> None:
    """Cap vision tokens for VRAM. / 限制视觉 token 以控制显存。"""
    ip = processor.image_processor
    min_px = min_pixels_factor * 28 * 28
    max_px = max_pixels_factor * 28 * 28
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
    """Load 4-bit Qwen3.5 and freeze the vision tower. / 4-bit 加载并冻结视觉塔。"""
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
    if hasattr(model, "visual"):
        for p in model.visual.parameters():
            p.requires_grad = False
        print("vision tower frozen (model.visual)", flush=True)
    return model, processor


def build_peft_model(model, init_adapter: str | None):
    """Attach a fresh spatial LoRA or resume one. / 新建或恢复 spatial LoRA。"""
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    if init_adapter:
        print(f"Resuming spatial adapter from {init_adapter}", flush=True)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial-grounding QLoRA trainer")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--init_adapter", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--max_pixels_factor", type=int, default=DEFAULT_MAX_PIXELS_FACTOR)
    parser.add_argument("--early_stopping_patience", type=int, default=2)
    args = parser.parse_args()

    use_early_stop = args.early_stopping_patience is not None and args.early_stopping_patience > 0
    model, processor = load_model_and_processor(max_pixels_factor=args.max_pixels_factor)
    model = build_peft_model(model, args.init_adapter)
    model.config.use_cache = False

    train_ds = SpatialGroundingDataset(args.train_jsonl)
    val_ds = SpatialGroundingDataset(args.val_jsonl) if args.val_jsonl else None
    collator = SpatialCollator(processor)
    print(f"train={len(train_ds)} val={len(val_ds) if val_ds else 0}", flush=True)
    if use_early_stop and not val_ds:
        raise ValueError("--early_stopping_patience requires --val_jsonl")

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
        save_strategy="epoch",
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
        print(f"early_stopping patience={args.early_stopping_patience}", flush=True)

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved spatial adapter -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
