# Stage II Module 2 — crash type and involved vehicles

RT-DETR tracks vehicles with ByteTrack on a short window around the onset, then a geometry rule plus a scene-layout rerank assigns one of five types: rear-end, head-on, T-bone, sideswipe, single-vehicle. The paper does not use CoTracker, so that runner is not included.

```bash
python run_intersection_rtdetrxl_bytetrack.py
python rerank_type_majority.py --help
```

`rtdetr-x.pt` is read from `$CRASHALIGN_DATA/rgbt_object_detection_310/`. It is not in this package.

The angle and overlap thresholds printed in paper Section 3.2 were chosen on the validation split. They are not stored as a separate constants file in the experiment tree. The runnable decision rule is `classify_crash_type` together with `apply_scene_reranking` in `run_intersection_rtdetrxl_bytetrack.py`.
