#!/usr/bin/env python3
"""
0917_Model helpers: 4-bit load, freeze policy, LoRA target regexes, LR groups.

Qwen3.5 VLM layout (important):
  model.visual.blocks[0..26]  — vision encoder (27 blocks)
  model.visual.merger         — multimodal projector (linear_fc1/fc2)
  model.language_model        — LLM (linear_attn + self_attn layers)

Under NF4, the projector is adapted with LoRA on merger linears (QLoRA-safe).
在 NF4 下用 merger 上的 LoRA 适配 projector（避免直接训练 4bit 全精度权重）。
"""

from __future__ import annotations

import os
import re
from typing import Iterable

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5ForConditionalGeneration

MODEL_ID = os.environ.get("MODEL_PATH", "Qwen/Qwen3.5-27B")
HF_CACHE = os.environ.get("HF_HOME", "/home/disk3/Jiachen/cache/huggingface")

DEFAULT_MAX_PIXELS_FACTOR = 256
DEFAULT_MIN_PIXELS_FACTOR = 128

# Last 4 vision blocks (27 total → indices 23..26).
# 视觉塔共 27 层，最后 4 层索引 23–26。
VISION_LAST_N = 4
VISION_N_BLOCKS = 27


def _vision_last_indices(n_last: int = VISION_LAST_N) -> list[int]:
    start = max(0, VISION_N_BLOCKS - n_last)
    return list(range(start, VISION_N_BLOCKS))


def build_target_regex(use_vision_lora: bool, n_last: int = VISION_LAST_N) -> str:
    """
    Regex for peft target_modules.
    Always: LLM attn/MLP + visual.merger (projector).
    Optional: last-N vision block linears.
    始终包含 LLM 注意力/MLP + merger；可选最后 N 层 vision LoRA。
    """
    llm = (
        r"model\.language_model\.layers\.\d+\."
        r"(?:"
        r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
        r"linear_attn\.(?:out_proj|in_proj_qkv)|"
        r"mlp\.(?:gate_proj|up_proj|down_proj)"
        r")"
    )
    merger = r"model\.visual\.merger\.linear_fc[12]"
    parts = [llm, merger]
    if use_vision_lora:
        idxs = _vision_last_indices(n_last)
        idx_alt = "|".join(str(i) for i in idxs)
        vis = (
            rf"model\.visual\.blocks\.(?:{idx_alt})\."
            r"(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])"
        )
        parts.append(vis)
    return r"(?:" + "|".join(parts) + r")"


def configure_processor_pixels(
    processor,
    max_pixels_factor: int = DEFAULT_MAX_PIXELS_FACTOR,
    min_pixels_factor: int = DEFAULT_MIN_PIXELS_FACTOR,
) -> None:
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


def get_visual(model):
    """Resolve vision tower under ForConditionalGeneration wrappers."""
    # 兼容 ForConditionalGeneration：视觉塔在 model.model.visual。
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "model") and hasattr(model.model, "visual"):
        return model.model.visual
    raise AttributeError("Cannot find model.visual")


def freeze_base(model) -> None:
    """
    Freeze all base params; LoRA / selected modules will re-enable grads later.
    先冻结全部基础权重；随后由 LoRA / 指定模块重新打开梯度。
    """
    for p in model.parameters():
        p.requires_grad = False
    vis = get_visual(model)
    print(
        f"froze base; vision blocks={len(vis.blocks)} merger={type(vis.merger).__name__}",
        flush=True,
    )


def load_model_and_processor(max_pixels_factor: int = DEFAULT_MAX_PIXELS_FACTOR):
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
    freeze_base(model)
    return model, processor


def attach_lora(
    model,
    use_vision_lora: bool,
    n_last: int = VISION_LAST_N,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    init_adapter: str | None = None,
):
    """Attach peft LoRA with config-specific target regex. / 按配置挂载 LoRA。"""
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # Re-assert freeze after prepare_model_for_kbit_training (it may enable embeds).
    # prepare_model_for_kbit_training 可能打开 embedding；再次冻结基础权重。
    freeze_base(model)

    if init_adapter:
        print(f"resume adapter {init_adapter}", flush=True)
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        regex = build_target_regex(use_vision_lora=use_vision_lora, n_last=n_last)
        print(f"LoRA target regex: {regex}", flush=True)
        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=regex,
        )
        model = get_peft_model(model, cfg)

    # Ensure only LoRA params (and any peft-marked) require grad.
    # 仅保留 LoRA（及 peft 标记）参数可训练。
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad = True
        elif p.requires_grad and "lora_" not in n:
            # keep frozen unless peft modules_to_save
            if "modules_to_save" not in n:
                p.requires_grad = False

    model.print_trainable_parameters()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable tensors={sum(p.requires_grad for p in model.parameters())} "
          f"params={n_train}", flush=True)
    return model


def param_groups(
    model,
    lr_llm: float,
    lr_projector: float,
    lr_vision: float,
) -> list[dict]:
    """
    Split LoRA params into LLM / projector(merger) / vision-last groups.
    将 LoRA 参数分为 LLM / projector(merger) / vision 末层三组学习率。
    """
    llm, proj, vis, other = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "visual.merger" in n:
            proj.append(p)
        elif re.search(r"visual\.blocks\.\d+", n):
            vis.append(p)
        elif "language_model" in n or "lm_head" in n:
            llm.append(p)
        else:
            other.append(p)
    groups = []
    if llm:
        groups.append({"params": llm, "lr": lr_llm, "name": "llm_lora"})
    if proj:
        groups.append({"params": proj, "lr": lr_projector, "name": "projector_lora"})
    if vis:
        groups.append({"params": vis, "lr": lr_vision, "name": "vision_lora"})
    if other:
        groups.append({"params": other, "lr": lr_llm, "name": "other"})
    for g in groups:
        n = sum(x.numel() for x in g["params"])
        print(f"param_group {g['name']}: n={n} lr={g['lr']}", flush=True)
    return groups


def classify_adapter_modules(model) -> dict[str, int]:
    """Count matched trainable LoRA modules by family (debug)."""
    counts = {"llm": 0, "merger": 0, "vision_block": 0}
    for n, p in model.named_parameters():
        if not p.requires_grad or "lora_" not in n:
            continue
        if "visual.merger" in n:
            counts["merger"] += 1
        elif "visual.blocks" in n:
            counts["vision_block"] += 1
        else:
            counts["llm"] += 1
    return counts
