# Paper figures for the README

Figures and tables are cropped from the manuscript. The running header and body text are removed. Captions below are the printed English captions.

Suggested README order:

1. `fig1_overview.png` — repository front page, full pipeline
2. `fig2_stage1_training.png` — how Stage I is trained
3. `table1_crash_recognition.png` — crash recognition results
4. `fig3_phase_onset.png` and `table2_onset_ablation.png` — onset
5. `fig4_spatial_intervention.png` — spatial localization
6. `fig5_pipeline_examples.png` — end-to-end examples
7. `table3_crash_understanding.png` and `fig6_robustness.png` — Stage II results and robustness

---

## Figure 1 — framework

Place this at the top of the README. It shows the five modules: recognition, onset, spatial localization, crash type, and the report.

![Figure 1. Overview of the CrashAlign framework.](docs/figures/fig1_overview.png)

```markdown
![Figure 1. Overview of the CrashAlign framework.](docs/figures/fig1_overview.png)
```

## Figure 2 — Stage I training

Module 1 is a window head followed by max pooling. Module 2 trains a four-phase classifier, freezes it, then trains the T head with a Gaussian MSE.

![Figure 2. Stage I training pipeline.](docs/figures/fig2_stage1_training.png)

```markdown
![Figure 2. Stage I training pipeline.](docs/figures/fig2_stage1_training.png)
```

## Table 1 — crash recognition

![Table 1. Comparison of crash recognition methods.](docs/figures/table1_crash_recognition.png)

```markdown
![Table 1. Comparison of crash recognition methods.](docs/figures/table1_crash_recognition.png)
```

## Figure 3 — phase-aware onset

![Figure 3. Visualization of phase-aware temporal onset localization.](docs/figures/fig3_phase_onset.png)

```markdown
![Figure 3. Visualization of phase-aware temporal onset localization.](docs/figures/fig3_phase_onset.png)
```

## Table 2 — onset ablation

![Table 2. Ablation study of temporal onset localization.](docs/figures/table2_onset_ablation.png)

```markdown
![Table 2. Ablation study of temporal onset localization.](docs/figures/table2_onset_ablation.png)
```

## Figure 4 — spatial localization

Fine-tuned Qwen3.5-27B compared with two zero-shot models. After the collision region is masked, the fine-tuned model no longer calls the frame a crash.

![Figure 4. Intervention-based analysis of VLM spatial localization.](docs/figures/fig4_spatial_intervention.png)

```markdown
![Figure 4. Intervention-based analysis of VLM spatial localization.](docs/figures/fig4_spatial_intervention.png)
```

## Figure 5 — full pipeline examples

The yellow box is the estimated onset frame. The yellow marker is the collision point. The top example is a nighttime T-bone. The bottom example is a rear-end crash in a tunnel.

![Figure 5. Examples of the crash detection and understanding pipeline.](docs/figures/fig5_pipeline_examples.png)

```markdown
![Figure 5. Examples of the crash detection and understanding pipeline.](docs/figures/fig5_pipeline_examples.png)
```

## Table 3 — crash understanding

![Table 3. Comparison of crash understanding models.](docs/figures/table3_crash_understanding.png)

```markdown
![Table 3. Comparison of crash understanding models.](docs/figures/table3_crash_understanding.png)
```

## Figure 6 — weather, scene, video quality

![Figure 6. Performance across weather, scene layout, and video quality.](docs/figures/fig6_robustness.png)

```markdown
![Figure 6. Performance across weather, scene layout, and video quality.](docs/figures/fig6_robustness.png)
```
