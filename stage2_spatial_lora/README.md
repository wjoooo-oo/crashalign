# Spatial QLoRA, language layers only

Control adapter: vision tower frozen, LoRA on the language projections. The paper's spatial module also adapts the merger and the last four vision blocks; that code is `../stage2_spatial_vlora/`.

对照适配器：视觉塔冻结，只训练语言层 LoRA。论文设定还训练 merger 和最后 4 个视觉 block，见 `../stage2_spatial_vlora/`。

# 0914_ Spatial Grounding QLoRA / 空间定位独立 LoRA

> Separate QLoRA adapter for **spatial collision localization**, independent from
> the crash-understanding LoRA in `0609crashexplanation/lora_finetune/`.
> 与事故语义理解 LoRA **分离**的空间定位适配器；预训练骨干与视觉编码器保持冻结。

## Design / 设计

| Item | Choice |
|------|--------|
| Base | `Qwen/Qwen3.5-27B` (4-bit NF4) |
| Adapter | QLoRA r=8, α=16, LM projections only |
| Vision | **Frozen** (`model.visual` + no LoRA on vision modules) |
| Input | **Same as `qwen35_spatial/`**: single predicted-onset frame |
| Target | GT `(center_x, center_y)` JSON |
| Crash LoRA | **Not** loaded / not mixed |

```
onset frame  →  frozen Qwen3.5 backbone  →  spatial LoRA  →  {"center_x","center_y","confidence"}
```

## Data split (200-GT, seed=42)

| Split | N | Ratio |
|-------|---|-------|
| train | 120 | 60% |
| val   | 40  | 20% |
| test  | 40  | 20% |

Frames = existing `test_200_GT/predicted_onset_frames/` (pipeline onset, not GT T).
Supervision = `ground_truth_200.csv` spatial points.

## Quick start

```bash
cd /home/disk3/Jiachen/accident/0914_spatial_lora
export PY=/home/disk3/Jiachen/envs/py310/bin/python

# 1) Build jsonl from onset index + GT
$PY prepare_data.py

# 2) Train spatial adapter (vision frozen)
bash run_train.sh

# 3) Zero-shot vs LoRA on holdout test40
$PY infer_eval.py --adapter runs/spatial_qlora --split test
```

## Metrics

Isotropic ACCIDENT spatial score (σ=0.1):

\[
\mathcal{S}=\exp\!\left(-\frac{d^2}{2\sigma^2}\right),\quad
d=\sqrt{(x-\hat x)^2+(y-\hat y)^2}
\]

Reports mean S, mean/median distance, and vs zero-shot on the **same** onset frames.

## Results (test40 holdout, σ=0.1)

Same predicted-onset frames as `qwen35_spatial/`; ZS vs spatial QLoRA (best val epoch≈2, early-stop).

| Method | S | mean dist | median dist | d&lt;0.10 |
|--------|---|-----------|-------------|---------|
| Zero-shot | **0.378** | 0.204 | 0.168 | 30% |
| Spatial QLoRA | **0.386** | 0.199 | 0.159 | 35% |
| Δ (LoRA−ZS) | **+0.008** | −0.004 | — | +5 pp |

Artifacts: `runs/eval/test_compare.json`, `test_*_preds.jsonl`, adapter `runs/spatial_qlora/`.

**Takeaway:** On this 200-GT holdout with pipeline onset frames, light QLoRA gives only a marginal S gain. Spatial accuracy remains bottlenecked more by onset quality / hard cases than by missing a spatial adapter. Crash-understanding LoRA stays a separate adapter.
