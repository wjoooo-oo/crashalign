# Stage II Module 2 — crash type and involved vehicles

RT-DETR tracks vehicles with ByteTrack on a short window around the onset, then a geometry rule plus a scene-layout rerank assigns one of five types: rear-end, head-on, T-bone, sideswipe, single-vehicle.

在 onset 附近的短窗口上用 RT-DETR + ByteTrack 得到轨迹，再由几何规则和场景重排得到碰撞类型。论文没有使用 CoTracker，因此对应脚本未收入。

```bash
python run_intersection_rtdetrxl_bytetrack.py
python rerank_type_majority.py --help
```

`rtdetr-x.pt` is read from `$CRASHALIGN_DATA/rgbt_object_detection_310/`. It is not in this package.

The angle and overlap thresholds printed in paper Section 3.2 were chosen on the validation split. They are not stored as a separate constants file in the experiment tree. The runnable decision rule is `classify_crash_type` together with `apply_scene_reranking` in `run_intersection_rtdetrxl_bytetrack.py`.
