# CrashAlign

Phase-aware, task-aligned traffic crash understanding from roadside video.
按论文 *Phase-Aware and Task-Aligned Traffic Crash Understanding from Surveillance Videos* 整理的代码包：只保留五模块主流程，不含中途探测实验。

```
Video
  │
  ▼
Stage I
  Module 1  Crash recognition     VideoMAE window head, video score = max over windows
  Module 2  Onset localization    VideoMAE → BiGRU → 4-phase head, then a frozen phase-conditioned T head
  │  onset time
  ▼
Stage II
  Module 1  Spatial localization  Qwen3.5-27B QLoRA (merger + LM + last 4 ViT blocks)
  Module 2  Crash type + vehicles RT-DETR + ByteTrack + geometry / scene rules
  Module 3  Crash report          separate Qwen3.5-27B QLoRA on language layers only
```

| Paper module | Code | Source experiment |
|---|---|---|
| I-1 Crash recognition | `stage1_crash_tonset/train_videomae_accident.py`, `0925_partial_finetune_encoder.py` | `accident/0723_task1_cumphase` |
| I-2 Onset | `stage1_crash_tonset/0728_train_phase_progression.py`, `0728_full_accident_phase_t.py`, `t_stage.py` | same |
| I precursor | `stage1_cumcrash/` | `accident/0722_cumcrash` |
| II-1 Spatial | `stage2_spatial_vlora/` (paper adapter), `stage2_spatial_lora/` (LM-only control) | `0917_spatial_vlora_ablation`, `0914_spatial_lora` |
| II-2 Type | `stage2_type/run_intersection_rtdetrxl_bytetrack.py` | accident root |
| II-3 Report | `stage2_report/lora_finetune/` | `accident/0609crashexplanation` |

Earlier package `1029_crashalign` is unchanged. It mixed zero-shot spatial scripts and internal ablations into Stage II, and it did not include the cumulative model in `0722_cumcrash`.

Paper figures and result tables, cropped for the README, are in [`1109_PAPER_FIGURES.md`](1109_PAPER_FIGURES.md).
论文图和结果表已裁好，写 README 时看 [`1109_PAPER_FIGURES.md`](1109_PAPER_FIGURES.md)。

## What this package keeps

Stage I recognition is a VideoMAE window encoder plus a clip score. `train_videomae_accident.py` scores a clip by the maximum window probability. `0925_partial_finetune_encoder.py` is the top-block fine-tune used before that encoder is plugged into the temporal head. Matched crash / non-crash clips (same source video, duration gap under 1 s, cut 2 s before onset) are built by `0726_build_accident_matched_split.py`.

Stage I onset is the Table 2 comparison, not a bag of diagnostics:

- `t_only`: Gaussian onset regression from VideoMAE features
- `phase_only`: onset read from the impact-phase peak
- `phase_cat_T`: freeze the phase model, concatenate the 4-class posterior, train the T head

Phase boundaries match the paper: normal before \(t-2.0\) s, pre-crash \([t-2.0, t-0.5)\), impact within \(\pm 0.5\) s, post-crash after \(t+0.5\) s (`labels.py`).

`stage1_cumcrash/` is the earlier monotone cumulative head (pre / post). The 4-phase model replaced it. It is here because that is the pipeline `0722_cumcrash` actually trained, not because the paper figure still uses it.

Stage II spatial follows the paper prompt: one onset frame, answer `x y` on a 0–1000 grid. The paper adapter puts LoRA on the multimodal merger, the language layers, and the last four vision blocks. That is `stage2_spatial_vlora/` (`run_vision_lora.sh`). `stage2_spatial_lora/` freezes the vision tower and adapts language layers only; keep it as the control, not as a second method.

Stage II type runs RT-DETR and ByteTrack on a short window around onset, then assigns rear-end, head-on, T-bone, sideswipe, or single-vehicle. The CoTracker runner is omitted. Section 3.2 of the paper lists the validation-chosen angle and overlap thresholds. Those constants are not a separate file in the experiment tree; the runnable classifier is the geometry plus scene rerank in `stage2_type/`.

Stage II report uses a second QLoRA adapter, vision frozen, language layers only. `lora_finetune/run_train.sh` is the two-step schedule: one epoch at \(1\times10^{-4}\), then a lower learning rate \(2\times10^{-5}\) with early stopping. Prompts that insert WHEN / WHERE / WHAT are in `stage2_report/prompts.py`.

Table 1 baselines that are part of the paper live in `stage1_crash_tonset/baselines_matched/` (stacked autoencoder, MDTS-ADNet, TCFF-GAN, Qwen3.5-27B zero-shot). TCFF-GAN code is under `third_party/so-tad/`. MDTS-ADNet still imports the external checkout `experiments_aap/sotad_tad`.

## Left out on purpose

- Embedding caches, frame caches, raw videos, and checkpoints
- Easy-split, reclip, smoke, FP-gate, shuffle, and ensemble probes
- CoTracker type runs and zero-shot spatial debugging scripts
- The internal Table 4 script in `accident/0925_e2e_settings`. Its clip count and scores are not the tables in the paper

## Data root

Videos and CSVs stay in the original tree. This package finds them through `CRASHALIGN_DATA`, or through a sibling directory named `accident`.

```bash
export CRASHALIGN_DATA=/path/to/accident
pip install -r requirements.txt
```

Weights are gitignored. Place VideoMAE, QLoRA adapters, and `rtdetr-x.pt` on the data tree or pass their paths to the scripts.

## Layout

```
1109_crashalign/
├── stage1_cumcrash/          # 0722 cumulative precursor
├── stage1_crash_tonset/      # recognition + phase-aware onset + Table 1 baselines
├── stage2_spatial_lora/      # LM-only spatial QLoRA
├── stage2_spatial_vlora/     # paper spatial QLoRA (merger + last-4 ViT)
├── stage2_type/              # RT-DETR + ByteTrack
├── stage2_report/            # report QLoRA
├── third_party/so-tad/       # TCFF-GAN
└── crashalign_paths.py
```
