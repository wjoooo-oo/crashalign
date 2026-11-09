# 0726 ACCIDENT-only matched benchmark

## Protocol

1. **Split by source video** (YouTube id, trailing `_NN` stripped) before any clip extraction.
2. **Fixed-length clips** with exactly `L` valid windows and **no padding** (`L=8` default → 3.75 s).
3. **Positive**: must contain the annotated onset; onset placement is **random** in training and **md5-hash deterministic** for val/test.
4. **Clean-normal negative**: same source mp4; last window center `<= onset - 3.0 s`; no overlap with the pre-crash safety region; all windows labeled `normal`.
5. **Phase labels** (window center time):
   - normal: `t < onset - 2.0`
   - pre-crash: `onset - 2.0 <= t < onset - 0.5`
   - impact: `|t - onset| <= 0.5`
   - post-crash: `t > onset + 0.5`
6. Videos without a valid same-video clean-normal are **excluded** from the matched benchmark and listed in `excluded.jsonl` / `excluded_final.jsonl`.

## Pipeline

```bash
# 1) Source-video split + prelim matched list
python 0726_build_accident_matched_split.py --fixed_len 8

# 2) Smoke (no GPU)
python 0726_smoke_test_accident_matched.py

# 3) Seed cache from existing ACCIDENT embeddings, then extract missing
python 0726_seed_cache_accident.py
python extract_embeddings.py --split train --data_dir data_accident_matched \
  --output_dir cache_emb_accident --crash_only
python extract_embeddings.py --split val   --data_dir data_accident_matched \
  --output_dir cache_emb_accident --crash_only
python extract_embeddings.py --split test  --data_dir data_accident_matched \
  --output_dir cache_emb_accident --crash_only

# 4) Finalize matched index on real window centers
python 0726_build_accident_matched_index.py --fixed_len 8

# 5) Train (example)
python train_task1.py \
  --dataset accident_matched \
  --matched_data_dir data_accident_matched \
  --cache_dir cache_emb_accident \
  --fixed_len 8 \
  --phase_mode softmax4 \
  --crash_feat full \
  --lambda_phase 1.0 \
  --output_dir checkpoints/accident_matched_full
```

## Val protocol check (after embedding extraction)

On the finalized val matched set (`L=8`):

- 75/75 prelim-matched videos have valid pos and neg placements on real window centers
- Dataset expands to 150 items (75 pos + 75 neg), each `emb.shape = [8, 768]`, `valid_mask` all-True
- Shortcut probes on these 150 clips:

| Probe | BA | AUROC |
| --- | ---: | ---: |
| Constant length (=L) | 0.500 | 0.500 |
| Single window | 0.460 | 0.145 |
| Mean-pooled windows | 0.460 | 0.124 |

Length is uninformative by construction; same-video negatives remove the
cross-collection appearance shortcut that broke ACCIDENT+SO-TAD.