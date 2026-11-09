# Task-1: crash/non-crash detection (TAD) + onset localization (T)

Started 2026-07-23. **Read `results/0729_stage1_summary.md` first** — it contains
every Stage-1 result, all ablations, and the reasoning behind the protocol choices.

## Results in one line

| Task | Best | Baseline |
|---|---|---|
| TAD (crash vs non-crash) | Balanced accuracy **0.602** (`h_only`, matched protocol) | near chance; phase does not help |
| T (onset localization) | MAE **2.324 s**, T-score **0.680** (`phase_cat_T`) | dedicated T head: 4.325 s / 0.550 |

## Protocol warning (read before adding any experiment)

**Never evaluate detection on a pool whose positives and negatives come from
different collections, or differ systematically in clip length.** Both shortcuts
fully saturate the task without any collision being present:

- ACCIDENT crash vs SO-TAD non-crash: BA 0.993 even when the ACCIDENT clips
  contain **no collision at all**. Bits-per-pixel alone separates the datasets
  at BA 1.000. Re-encoding to a common format does not fix it.
- SOTAD-IID: crash clips were systematically shorter, so clip length alone gives
  BA 0.999.

Use the **ACCIDENT matched protocol**, where positive and negative clips come from
the *same source video* at fixed length. It reports 0.60 rather than 0.99, and
0.60 is the real number. Evidence: `results/dataset_confusion/`,
`results/dataset_confound/`, `results/dataset_gap/`, `results/shortcut_controls/`.

## Model

```
2 s / 0.25 s windows → frozen VideoMAE z_t → BiGRU h_t
  → phase posterior over {normal, pre_impact, impact, post_impact}
  → crash MLP (feature and/or phase channels; see crash_feat)
  → onset score head (T stage), optionally conditioned on the phase posterior
```

Phase boundaries come from `0720_phase_tad` and live in `labels.py`.

## Layout

Libraries (importable, no date prefix): `model.py`, `labels.py`, `losses.py`,
`sampling.py`, `t_stage.py` (shared onset-stage dataset/head/decoder/metrics).

Dated scripts are one experiment each; `results/0729_stage1_summary.md` §5 maps
every script to the result it produced.

Data: `data_accident_matched/` + `cache_emb_accident/` (matched TAD),
`cache_emb_accident_full/` (all 2027 crash videos, for T).

## Reproduce the two headline results

```bash
cd /home/disk3/Jiachen/accident/0723_task1_cumphase
PY=/home/jiachen/.conda/envs/traffix/bin/python

$PY 0726_smoke_test_accident_matched.py          # protocol sanity checks

# TAD: matched-protocol head ablation
bash 0726_run_accident_matched_ablation.sh

# T: full ACCIDENT, phase + onset head
$PY 0728_full_accident_phase_t.py
```
