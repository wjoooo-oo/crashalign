# Stage I — crash recognition and onset localization

Paper Stage I. Library files (`model.py`, `labels.py`, `losses.py`, `sampling.py`, `t_stage.py`) are shared by both modules. Both modules use the same VideoMAE window features.

## Module 1 — crash recognition

Matched ACCIDENT+ clips: one crash clip and one non-crash clip from the same source video.

```bash
python 0726_build_accident_matched_split.py
python 0726_build_accident_matched_index.py
python train_videomae_accident.py          # window scores, clip score = max
python 0925_partial_finetune_encoder.py    # top-block fine-tune used by the reported detector
python 0726_eval_accident_matched.py
```

Table 1 baselines:

```bash
python baselines_matched/run_stacked_ae.py
python baselines_matched/run_mdts_adnet.py      # needs experiments_aap/sotad_tad
python baselines_matched/run_tcff_gan.py        # uses ../../third_party/so-tad
python baselines_matched/run_qwen_zero_shot.py
```

## Module 2 — phase-aware onset

Phase labels are in `labels.py` (normal / pre-crash / impact / post-crash).
`0728_full_accident_phase_t.py` trains the three Table 2 rows: direct T head, phase-peak decoding, and the phase-conditioned T head. The phase head itself is `0728_train_phase_progression.py`. The T head, Gaussian target, and stop-gradient phase features are in `t_stage.py`.

```bash
python 0728_extract_accident_full.py
python 0728_full_accident_phase_t.py
python 0727_eval_accident_t_complete.py
```

`train_task1.py` trains the BiGRU phase model and the crash MLP on cached embeddings. That is the sequence model behind Module 2, and the crash MLP used when recognition reads the BiGRU state.

Not included: easy-split selection, reclip, smoke tests, FP-gate, order-shuffle probes, and ensemble scripts. Those were intermediate checks.
