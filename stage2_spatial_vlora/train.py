#!/usr/bin/env python3
"""
0917_Train spatial VLoRA ablation (main vs vision_lora).

Loss: token-level CE on answer tokens only
  L_M1 = -1/N Σ log P(y_n | y_<n, I, q)

Checkpoint selection uses validation spatial score S (σ=0.1), not CE.
用验证集空间分 S 选 checkpoint，而不是 CE。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments

HERE = Path(__file__).resolve().parent
from dataset import COORD_SCALE, SPATIAL_INSTRUCTION, SpatialCollator, SpatialIntDataset  # noqa: E402
from model_utils import (  # noqa: E402
    DEFAULT_MAX_PIXELS_FACTOR,
    VISION_LAST_N,
    attach_lora,
    classify_adapter_modules,
    load_model_and_processor,
    param_groups,
)

SIGMA = 0.1


def spatial_score(x_gt: float, y_gt: float, x_pred: float, y_pred: float, sigma: float = SIGMA) -> float:
    dx = float(x_gt) - float(x_pred)
    dy = float(y_gt) - float(y_pred)
    return math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))


def parse_xy(text: str) -> tuple[float, float] | None:
    """Parse 'x y' integers or JSON-ish floats back to normalized coords."""
    # 解析整数 "x y" 或 JSON 浮点，转回归一化坐标
    clean = text
    if "</think>" in clean:
        clean = clean.split("</think>")[-1]
    clean = clean.strip()
    m = re.search(r"(-?\d+)\s*[,\s]\s*(-?\d+)", clean)
    if m:
        xi, yi = int(m.group(1)), int(m.group(2))
        return max(0, min(COORD_SCALE, xi)) / COORD_SCALE, max(0, min(COORD_SCALE, yi)) / COORD_SCALE
    m2 = re.search(r'"center_x"\s*:\s*([\d.]+).*?"center_y"\s*:\s*([\d.]+)', clean, re.DOTALL)
    if m2:
        return float(m2.group(1)), float(m2.group(2))
    return None


@torch.no_grad()
def eval_spatial_s(model, processor, records: list[dict[str, Any]], max_new_tokens: int = 16) -> dict[str, float]:
    """Generate on val/test records and compute mean S. / 生成并计算平均 S。"""
    from PIL import Image

    model.eval()
    scores, dists = [], []
    for rec in records:
        img = Image.open(rec["frames"][0]).convert("RGB")
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
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        raw = processor.batch_decode([out[0][in_len:]], skip_special_tokens=True)[0].strip()
        xy = parse_xy(raw)
        if xy is None:
            xy = (0.5, 0.5)
        s = spatial_score(rec["target"]["center_x"], rec["target"]["center_y"], xy[0], xy[1])
        d = math.hypot(rec["target"]["center_x"] - xy[0], rec["target"]["center_y"] - xy[1])
        scores.append(s)
        dists.append(d)
    return {
        "S_mean": float(sum(scores) / max(len(scores), 1)),
        "dist_mean": float(sum(dists) / max(len(dists), 1)),
        "n": float(len(scores)),
    }


class SpatialTrainer(Trainer):
    """Trainer with custom LR groups. / 带分组学习率的 Trainer。"""

    def __init__(self, *args, optim_groups: list[dict] | None = None, **kwargs):
        self._optim_groups = optim_groups
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        if self.optimizer is None and self._optim_groups is not None:
            # Drop helper key before AdamW.
            # 去掉辅助 name 字段再交给 AdamW。
            groups = [{"params": g["params"], "lr": g["lr"]} for g in self._optim_groups]
            self.optimizer = AdamW(groups, weight_decay=0.0)
            return self.optimizer
        return super().create_optimizer()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["main", "vision_lora"], required=True)
    ap.add_argument("--train_jsonl", default=str(HERE / "data" / "train.jsonl"))
    ap.add_argument("--val_jsonl", default=str(HERE / "data" / "val.jsonl"))
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr_llm", type=float, default=2e-5)
    ap.add_argument("--lr_projector", type=float, default=1e-5)
    ap.add_argument("--lr_vision", type=float, default=5e-6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--max_pixels_factor", type=int, default=DEFAULT_MAX_PIXELS_FACTOR)
    ap.add_argument("--vision_last_n", type=int, default=VISION_LAST_N)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    use_vision = args.config == "vision_lora"
    out_dir = Path(args.output_dir or (HERE / "runs" / args.config))
    out_dir.mkdir(parents=True, exist_ok=True)

    model, processor = load_model_and_processor(max_pixels_factor=args.max_pixels_factor)
    model = attach_lora(
        model,
        use_vision_lora=use_vision,
        n_last=args.vision_last_n,
    )
    print("adapter families:", classify_adapter_modules(model), flush=True)
    model.config.use_cache = False

    train_ds = SpatialIntDataset(args.train_jsonl)
    val_records = load_jsonl(Path(args.val_jsonl))
    collator = SpatialCollator(processor)
    groups = param_groups(model, args.lr_llm, args.lr_projector, args.lr_vision)

    # Manual epoch loop: train 1 epoch → eval S → early stop / save best.
    # 手动按 epoch：训练 1 轮 → 评 S → 早停 / 存最优。
    targs = TrainingArguments(
        output_dir=str(out_dir / "hf_trainer"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr_llm,  # overridden by param groups
        num_train_epochs=1,  # we loop externally
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        save_strategy="no",
        eval_strategy="no",
        optim="adamw_torch",
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
        dataloader_num_workers=0,
        seed=args.seed,
    )

    history = []
    best_s = -1.0
    best_epoch = -1
    stall = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== epoch {epoch}/{args.epochs} config={args.config} =====", flush=True)
        trainer = SpatialTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            data_collator=collator,
            optim_groups=groups,
        )
        trainer.train()
        # Refresh param groups next epoch (new optimizer); keep weights.
        # 下一 epoch 重建 optimizer；权重保留。
        model = trainer.model

        t0 = time.time()
        metrics = eval_spatial_s(model, processor, val_records)
        metrics["epoch"] = epoch
        metrics["seconds_eval"] = round(time.time() - t0, 1)
        history.append(metrics)
        print(f"val metrics: {metrics}", flush=True)

        if metrics["S_mean"] > best_s + 1e-4:
            best_s = metrics["S_mean"]
            best_epoch = epoch
            stall = 0
            model.save_pretrained(out_dir)
            processor.save_pretrained(out_dir)
            (out_dir / "best_meta.json").write_text(
                json.dumps({"best_epoch": best_epoch, "best_S": best_s, "config": args.config}, indent=2),
                encoding="utf-8",
            )
            print(f"saved best -> {out_dir} (S={best_s:.4f})", flush=True)
        else:
            stall += 1
            print(f"no improvement (stall={stall}/{args.patience})", flush=True)
            if stall >= args.patience:
                print("early stop on val S", flush=True)
                break

        if args.max_steps > 0:
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"done. best_epoch={best_epoch} best_S={best_s:.4f}", flush=True)


if __name__ == "__main__":
    main()
