# Stage II Module 1 — spatial QLoRA (paper setting)

This is the spatial module in the paper: Qwen3.5-27B, 4-bit QLoRA, \(r=8\), \(\alpha=16\), one GT onset frame, answer `x y` on a 0–1000 grid. `run_vision_lora.sh` adapts the merger, the language layers, and the last four vision blocks. `run_main.sh` freezes the vision blocks and is the control.

The notes below are the original 200-GT holdout log. They are not a second method. The paper reports the spatial score on the recognition-gated test clips, not this 40-clip holdout.

# 0917_ Spatial VLoRA (original log)

> Holdout log from the 200-GT split.

## Setup

| Item | Choice |
|------|--------|
| Base | `Qwen/Qwen3.5-27B` 4-bit NF4 |
| Input | **GT-annotated** onset frame + fixed prompt |
| Target | Integer coords on a 0–1000 grid (`x y`) |
| Freeze | Base LLM weights; most of vision encoder |
| Train | Multimodal projector (`visual.merger`) via LoRA; LLM attention(+MLP) LoRA |
| Ablation | + LoRA on vision `blocks[23:27]` (last 4) |
| Loss | Token CE on answer tokens only (labels −100 on image/prompt) |
| Select ckpt | Validation **spatial score S** (σ=0.1), not CE loss |

### Configs

1. **main** — frozen vision blocks + merger LoRA + LLM LoRA  
2. **vision_lora** — main + last-4-block vision LoRA  

### Hyperparams (defaults)

- \(r=8,\alpha=16\), dropout 0.05  
- LR: LLM LoRA \(2\times10^{-5}\); projector/merger \(1\times10^{-5}\); vision LoRA \(5\times10^{-6}\)  
- Epochs up to 10, early stop on val S (patience 2)  
- Same 200-GT split as `0914_spatial_lora` (120/40/40, seed=42)

## Notes

- Qwen3.5 projector ≡ `model.visual.merger` (`linear_fc1` / `linear_fc2`).
- Under 4-bit QLoRA, projector is adapted with LoRA on merger linears (full-precision FT of 4-bit weights is not used).
- Vision path is `model.visual` (not top-level `model.visual` on the ForCausal wrapper).

## Run

```bash
cd /home/disk3/Jiachen/accident/0917_spatial_vlora_ablation
export PY=/home/disk3/Jiachen/envs/py310/bin/python
$PY prepare_data.py
bash run_main.sh          # config main
bash run_vision_lora.sh   # config vision_lora
$PY infer_eval.py --split test
$PY infer_eval.py --adapter runs/main --split test
$PY infer_eval.py --adapter runs/vision_lora --split test
```

## Results (test40, GT onset, σ=0.1)

Both configs early-stopped at epoch 1 on val S (then degraded).

| Method | Val S (best) | Test S | Δ vs ZS |
|--------|--------------|--------|---------|
| Zero-shot | — | **0.426** | — |
| main (merger+LLM LoRA) | 0.496 | **0.396** | −0.030 |
| vision_lora (+ last-4) | 0.494 | **0.434** | +0.009 |

Artifacts: `runs/eval/test_compare.json`.

**Takeaway (trial):** On this small 200-GT split, adding last-4 vision LoRA does **not** clearly beat zero-shot on test (+0.009 only). Main without vision LoRA **hurts** test S. Treat as negative/weak ablation evidence; do not promote as main result.