# Stage I precursor — cumulative crash head

From `accident/0722_cumcrash`. This is the pipeline that predates the paper's 4-phase onset model. It is not the final structure in Figure 2.

A frozen VideoMAE encodes 2 s windows at a 0.25 s stride. A unidirectional GRU then emits monotone cumulative pre-crash and post-crash scores, and a video-level crash head reads that trajectory (`model.py`, `targets.py`).

```bash
python extract_embeddings.py --split train
python train_task1.py
python eval_task1.py
```

The paper's recognition and onset code is `../stage1_crash_tonset/`.
