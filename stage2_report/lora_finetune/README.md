# 0625_ Qwen3.5 Crash-Report LoRA Fine-tuning / 事故报告 LoRA 微调

> Last updated: 2025-06-25

Two-stage pipeline on ACCIDENT (~2000 clips):

1. **TSC (when / where / what)** — sub5 `submission5_qwen35_spatial.csv`
2. **LoRA description** — 20 frames + instruction → structured JSON report

---

## Data split / 数据划分（5:2:3）

Gold (200) + Silver (1827) pooled, shuffled (seed=42):

| Split | Size | Ratio |
|-------|------|-------|
| **train** | 1000 | 50% |
| **val** | 400 | 20% |
| **test** | 600 | 30% |
| held-out | 27 | — |

- Fixed fields in training targets = **official GT** (`ground_truth_full.csv` + `test_metadata_full.csv`)
- LoRA **inference input** = frames + instruction only (no sub5 metadata leaked)

---

## Frame strategy / 抽帧策略

**`pre13_post7`** — 20 frames: 13 pre-onset + 7 post-onset (no t=0 frame).

| Setting | Onset center | Cache dir | Test jsonl |
|---------|--------------|-----------|------------|
| Training / oracle eval | GT `accident_time` | `frames_cache_pre13_post7/` | `test.jsonl` |
| Deployment-realistic eval | sub5 predicted T | `frames_cache_pre13_post7_sub5/` | `test_sub5.jsonl` |

```bash
# Rebuild test set with sub5 onset (600 videos)
python prepare_test_sub5_frames.py
```

---

## Training / 训练

```
Stage 1: train.jsonl (1000 mixed)      @ lr 1e-4, 1 epoch  → runs/stage1_mixed
Stage 2: train_gold.jsonl (~95 gold)  @ lr 2e-5, up to 10 epochs, early-stop patience=2
         val_gold.jsonl (~40 gold)     → best: epoch 8, runs/stage2_gold_refine
```

- 4-bit QLoRA, r=8, frozen vision, `max_pixels_factor=128` (A6000 49GB)

---

## ACCIDENT TSC metrics (arXiv:2604.09819, Sec. 5)

Reference: [ACCIDENT benchmark paper](https://arxiv.org/abs/2604.09819).

### Temporal localization \(\mathcal{T}\) (Eq. 1–2)

Per video:

\[
\mathcal{T}_i = \exp\!\left(-\frac{(t^{*}_i - \hat{t}_i)^2}{2\sigma_t^2}\right)
\]

Dataset mean: \(\mathcal{T} = \frac{1}{N}\sum_i \mathcal{T}_i\).  
Default \(\sigma_t = 2.0\) s (paper also reports \(\sigma_t \in \{0.5, 1, 2\}\)).

### Spatial localization \(\mathcal{S}\) (Eq. 3–4)

**Paper (anisotropic):**

\[
\mathcal{S}_i = \exp\!\left(-\left[\frac{(x^{*}_i-\hat{x}_i)^2}{2\sigma_x^2}+\frac{(y^{*}_i-\hat{y}_i)^2}{2\sigma_y^2}\right]\right)
\]

\(\sigma_x, \sigma_y\) = mean normalized width/height of accident bounding boxes.

**Competition / sub5 (isotropic fallback):** when box sizes are unavailable, use \(\sigma_x=\sigma_y=\sigma=0.1\):

\[
\mathcal{S}_i = \exp\!\left(-\frac{d_i^2}{2\sigma^2}\right), \quad d_i = \sqrt{(x^{*}-\hat{x})^2+(y^{*}-\hat{y})^2}
\]

### Collision type \(\mathcal{C}\) (Eq. 5)

Top-1 accuracy:

\[
\mathcal{C} = \frac{1}{N}\sum_i \mathbb{1}[\hat{y}_i = y^{*}_i]
\]

### Unified score \(\mathrm{ACC}^{S}\) (Eq. 6)

Harmonic mean (penalizes weak dimensions):

\[
\mathrm{ACC}^{S} = \frac{3}{\dfrac{1}{\mathcal{T}} + \dfrac{1}{\mathcal{S}} + \dfrac{1}{\mathcal{C}}}
\]

```bash
# sub5 TSC on our 600-video test split (isotropic σ=0.1, σ_t=2.0)
python eval_tsc.py --out runs/stage2_gold_refine/tsc_sub5_test600.json
```

---

## Description metrics (LoRA output)

Evaluated on `preds_*.jsonl` via `eval_preds.py`:

| Metric | Definition |
|--------|------------|
| `parse_rate` | Valid JSON extracted |
| `scene/weather/lighting_acc` | Exact match vs official GT target |
| `context_mean` | Mean of scene + weather + lighting acc |
| `collision_type_acc` | Exact match vs GT type |
| `participants_f1` | Set F1 on `participants[]` vs target |
| `reasoning_coverage` | observation, explanation, summary all non-empty |

```bash
# Oracle onset (GT-centered frames)
python infer.py --adapter runs/stage2_gold_refine --test_jsonl data/test.jsonl \
    --out runs/stage2_gold_refine/preds_test.jsonl
python eval_preds.py --preds runs/stage2_gold_refine/preds_test.jsonl \
    --out runs/stage2_gold_refine/eval_test_gt_onset.json

# Deployment-realistic (sub5-centered frames)
python infer.py --adapter runs/stage2_gold_refine --test_jsonl data/test_sub5.jsonl \
    --out runs/stage2_gold_refine/preds_test_sub5_onset.jsonl
python eval_preds.py --preds runs/stage2_gold_refine/preds_test_sub5_onset.jsonl \
    --out runs/stage2_gold_refine/eval_test_sub5_onset.json
```

---

## Results (test n=600)

### TSC — sub5 vs GT (`submission5_qwen35_spatial.csv`, σ_t=2, σ=0.1)

| Metric | Score |
|--------|-------|
| \(\mathcal{T}\) | 0.688 |
| \(\mathcal{S}\) | 0.451 |
| \(\mathcal{C}\) | 0.575 |
| \(\mathrm{ACC}^{S}\) | 0.555 |
| \|Δt\| MAE | 2.99 s |
| \|Δt\| > 2 s | 30.8% |

### LoRA description — GT onset frames (`stage2_gold_refine`, `preds_test.jsonl`)

| Metric | All (600) | Gold (60) | Silver (540) |
|--------|-----------|-----------|--------------|
| parse_rate | 1.000 | 1.000 | 1.000 |
| context_mean | 0.920 | 0.928 | 0.919 |
| scene_acc | 0.843 | 0.867 | 0.841 |
| lighting_acc | 0.972 | 0.967 | 0.972 |
| weather_acc | 0.945 | 0.950 | 0.944 |
| collision_type_acc | 0.575 | 0.617 | 0.570 |
| participants_f1 | 0.886 | 0.843 | 0.890 |
| reasoning_coverage | 1.000 | 1.000 | 1.000 |

### LoRA description — sub5 onset frames

Run `prepare_test_sub5_frames.py` → `infer.py` on `test_sub5.jsonl` → `eval_preds.py`.  
Results: `runs/stage2_gold_refine/eval_test_sub5_onset.json` (pending after inference).

---

## Layout / 目录

```
lora_finetune/
  prepare_data.py              prepare_test_sub5_frames.py
  accident_metrics.py          eval_tsc.py   eval_preds.py
  dataset.py   train_lora.py   infer.py
  run_all.sh   run_train.sh   README.md
  data/
    train.jsonl  val.jsonl  test.jsonl  test_sub5.jsonl
    frames_cache_pre13_post7/  frames_cache_pre13_post7_sub5/
  runs/stage2_gold_refine/
    preds_test.jsonl
    preds_test_sub5_onset.jsonl   (after sub5 infer)
    tsc_sub5_test600.json
    eval_test_gt_onset.json
    eval_test_sub5_onset.json
```

---

## Hardware

Single RTX A6000 (49GB): 4-bit QLoRA + frozen vision + 20 frames + `max_pixels_factor=128`.
