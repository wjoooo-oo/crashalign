#!/usr/bin/env python3
"""
200GT intersection videos: RT-DETR-XL + ByteTrack (same as crash_localization Step 4),
then rule-based crash type + scene reranking. Compare C vs Sub5 (RT-DETR-L + ByteTrack).

Scene reranking (signalized / simple intersection) only remaps:
  single, sideswipe, head-on -> t-bone
It does NOT remap rear-end -> t-bone, so a wrong rear-end stays rear-end unless you add a rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crashalign_paths import accident_root  # noqa: E402

# Detector weights and 200-GT tables stay in the data tree.
# 检测器权重和 200-GT 表在数据目录。
_DATA = accident_root()
VIDEO_ROOT = _DATA / "test_videos" / "videos"
GT_CSV = _DATA / "test_200_GT" / "ground_truth_200.csv"
META_CSV = _DATA / "test_200_GT" / "test_metadata_200.csv"
RTDETR_XL_WEIGHTS = _DATA / "rgbt_object_detection_310" / "rtdetr-x.pt"
OUTPUT_DIR = _DATA / "intersection_rtdetrxl_bytetrack"

INTERSECTION_LAYOUTS = {"signalized_intersection", "simple_intersection", "grade_separated_intersection"}
VEHICLE_COCO_IDS = {2, 3, 5, 7}
VEHICLE_NAMES = {
    "car", "truck", "van", "bus", "vehicle", "automobile",
    "motorcycle", "motorbike", "motor", "pickup",
}

# Aligned with test_metadata/rerank_type_majority.py (no head-on -> t-bone at intersections).
SCENE_CORRECTIONS = {
    "signalized_intersection": {"single": "t-bone", "sideswipe": "t-bone"},
    "simple_intersection": {"single": "t-bone", "sideswipe": "t-bone"},
    "grade_separated_intersection": {},
}


@dataclass
class Detection:
    frame_idx: int
    track_id: int
    cls_name: str
    bbox: Tuple[float, float, float, float]
    score: float


@dataclass
class VehicleTrajectory:
    track_id: int
    cls_name: str
    frame_indices: List[int] = field(default_factory=list)
    centroids: List[Tuple[float, float]] = field(default_factory=list)
    speeds: List[float] = field(default_factory=list)
    directions: List[float] = field(default_factory=list)


@dataclass
class CollisionEvent:
    frame_idx: int
    track_id_a: int
    track_id_b: int
    iou: float
    iou_delta: float
    closing_speed: float = 0.0
    severity: float = 0.0


def _is_vehicle_cls_id(cls_id: int, names: Dict[int, str]) -> bool:
    if cls_id in VEHICLE_COCO_IDS:
        return True
    cls_name = names.get(int(cls_id), "")
    return cls_name.lower() in VEHICLE_NAMES


def load_gt_and_meta():
    with open(GT_CSV, newline="") as f:
        gt = {r["path"]: r for r in csv.DictReader(f)}
    with open(META_CSV, newline="") as f:
        meta = {r["path"]: r for r in csv.DictReader(f)}
    return gt, meta


def get_intersection_videos(gt, meta):
    videos = []
    for path, g in gt.items():
        sl = g.get("scene_layout", "").strip()
        if not sl:
            m = meta.get(path, {})
            sl = m.get("scene_layout", "").strip()
        if sl in INTERSECTION_LAYOUTS:
            stem = Path(path).stem
            video_file = VIDEO_ROOT / f"{stem}.mp4"
            if video_file.exists():
                fps_val = 30.0
                m = meta.get(path, {})
                try:
                    nf = float(m.get("no_frames", 0) or 0)
                    dur = float(m.get("duration", 0) or 0)
                    if nf > 0 and dur > 0:
                        fps_val = nf / dur
                except ValueError:
                    pass
                videos.append({
                    "path": path,
                    "stem": stem,
                    "video_file": video_file,
                    "gt_type": g["type"].strip().lower(),
                    "scene_layout": sl,
                    "fps": fps_val,
                })
    return videos


def decode_video(video_path: Path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def load_rtdetr(weights: Path):
    from ultralytics import RTDETR
    return RTDETR(str(weights))


def run_rtdetr_bytetrack(
    model,
    frames: List,
    frame_indices: List[int],
    conf: float = 0.25,
    iou: float = 0.45,
    imgsz: int = 640,
    tracker: str = "bytetrack.yaml",
) -> List[Detection]:
    """Same as crash_localization.detector_tracker.run_detection_tracking."""
    names = model.names
    all_detections: List[Detection] = []
    model.predictor = None

    for frame, fidx in tqdm(
        zip(frames, frame_indices), total=len(frames),
        desc="  RT-DETR-XL + ByteTrack", unit="frame", leave=False,
    ):
        results = model.track(
            frame, imgsz=imgsz, conf=conf, iou=iou,
            tracker=tracker, persist=True, verbose=False,
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        scores = r.boxes.conf.cpu().numpy()
        track_ids = (
            r.boxes.id.cpu().numpy().astype(int)
            if r.boxes.id is not None
            else np.arange(len(boxes))
        )

        for box, cls_id, score, tid in zip(boxes, cls_ids, scores, track_ids):
            cls_name = names.get(int(cls_id), str(cls_id))
            if not _is_vehicle_cls_id(int(cls_id), names):
                continue
            all_detections.append(Detection(
                frame_idx=fidx,
                track_id=int(tid),
                cls_name=cls_name,
                bbox=tuple(float(v) for v in box),
                score=float(score),
            ))

    return all_detections


def build_trajectories(detections):
    grouped = defaultdict(list)
    for det in detections:
        grouped[det.track_id].append(det)

    trajectories = {}
    for tid, dets in grouped.items():
        dets_sorted = sorted(dets, key=lambda d: d.frame_idx)
        traj = VehicleTrajectory(track_id=tid, cls_name=dets_sorted[0].cls_name)
        for det in dets_sorted:
            cx = (det.bbox[0] + det.bbox[2]) / 2.0
            cy = (det.bbox[1] + det.bbox[3]) / 2.0
            traj.frame_indices.append(det.frame_idx)
            traj.centroids.append((cx, cy))

        for i in range(len(traj.centroids)):
            if i == 0:
                traj.speeds.append(0.0)
                traj.directions.append(0.0)
            else:
                dx = traj.centroids[i][0] - traj.centroids[i - 1][0]
                dy = traj.centroids[i][1] - traj.centroids[i - 1][1]
                traj.speeds.append(float(np.sqrt(dx * dx + dy * dy)))
                traj.directions.append(float(np.arctan2(dy, dx)))
        trajectories[tid] = traj
    return trajectories


def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def detect_collisions(detections, trajectories, iou_threshold=0.10, delta_threshold=0.08):
    by_frame = defaultdict(list)
    for det in detections:
        by_frame[det.frame_idx].append(det)

    prev_iou = {}
    prev_dist = {}
    events = []

    for fidx in sorted(by_frame.keys()):
        frame_dets = by_frame[fidx]
        curr_iou = {}
        curr_dist = {}

        for da, db in combinations(frame_dets, 2):
            pair = (min(da.track_id, db.track_id), max(da.track_id, db.track_id))
            val = _iou(da.bbox, db.bbox)
            curr_iou[pair] = val

            ca = _centroid(da.bbox)
            cb = _centroid(db.bbox)
            dist = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
            curr_dist[pair] = dist

            prev_val = prev_iou.get(pair, 0.0)
            delta = val - prev_val

            closing = 0.0
            pd = prev_dist.get(pair)
            if pd and pd > 0:
                closing = max(0.0, (pd - dist) / pd)

            if val >= iou_threshold and delta >= delta_threshold:
                severity = val * (1.0 + delta) + 0.3 * closing
                events.append(CollisionEvent(
                    frame_idx=fidx, track_id_a=pair[0], track_id_b=pair[1],
                    iou=val, iou_delta=delta, closing_speed=closing, severity=severity,
                ))

        prev_iou = curr_iou
        prev_dist = curr_dist

    return events


def detect_anomalies(trajectories, z_speed=3.0, min_track_len=4, min_speed=2.0):
    anomalies = []
    for tid, traj in trajectories.items():
        n = len(traj.speeds)
        if n < min_track_len:
            continue
        speeds = np.array(traj.speeds[1:], dtype=np.float32)
        if speeds.size < 3:
            continue
        med = float(np.median(speeds))
        mad = float(np.median(np.abs(speeds - med)))
        scale = 1.4826 * mad + 1e-6
        for i in range(1, n):
            s = traj.speeds[i]
            z = (s - med) / scale
            if z > z_speed and s > min_speed:
                anomalies.append({"track_id": tid, "frame_idx": traj.frame_indices[i]})
    return anomalies


def classify_crash_type(collision_events, trajectories, anomalies):
    if not collision_events:
        return "single"

    best = max(collision_events, key=lambda e: e.severity)
    tid_a, tid_b = best.track_id_a, best.track_id_b
    fidx = best.frame_idx

    dir_a = _get_dir(trajectories.get(tid_a), fidx)
    dir_b = _get_dir(trajectories.get(tid_b), fidx)

    if dir_a is None or dir_b is None:
        return "rear-end" if best.iou > 0.3 else "sideswipe"

    angle = abs(dir_a - dir_b)
    angle = min(angle, 2 * math.pi - angle)
    deg = math.degrees(angle)

    if deg > 140:
        return "head-on"
    if 60 < deg < 120:
        return "t-bone"
    if deg < 30:
        return "rear-end"
    return "sideswipe"


def _get_dir(traj, fidx):
    if traj is None:
        return None
    for i, f in enumerate(traj.frame_indices):
        if f == fidx and i < len(traj.directions):
            return traj.directions[i]
    if not traj.frame_indices:
        return None
    closest = min(range(len(traj.frame_indices)), key=lambda i: abs(traj.frame_indices[i] - fidx))
    return traj.directions[closest] if closest < len(traj.directions) else None


def apply_scene_reranking(pred_type, scene_layout):
    corrections = SCENE_CORRECTIONS.get(scene_layout, {})
    return corrections.get(pred_type, pred_type)


def process_video(vinfo, model, device: str):
    frames = decode_video(vinfo["video_file"])
    if not frames:
        return {
            "stem": vinfo["stem"], "raw_type": "single", "pred_type": "single",
            "n_dets": 0, "n_tracks": 0, "n_coll": 0,
        }

    frame_indices = list(range(len(frames)))
    detections = run_rtdetr_bytetrack(model, frames, frame_indices)

    trajectories = build_trajectories(detections)
    collisions = detect_collisions(detections, trajectories)
    anomalies = detect_anomalies(trajectories)

    raw_type = classify_crash_type(collisions, trajectories, anomalies)
    final_type = apply_scene_reranking(raw_type, vinfo["scene_layout"])
    n_tracks = len(set(d.track_id for d in detections))

    return {
        "stem": vinfo["stem"],
        "raw_type": raw_type,
        "pred_type": final_type,
        "n_dets": len(detections),
        "n_tracks": n_tracks,
        "n_coll": len(collisions),
        "n_anom": len(anomalies),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    gt, meta = load_gt_and_meta()
    videos = get_intersection_videos(gt, meta)
    print(f"Intersection clips in 200GT (with local mp4): {len(videos)}")

    if args.limit:
        videos = videos[: args.limit]
        print(f"  limited to {args.limit}")

    print(f"\nLoading {RTDETR_XL_WEIGHTS} ...")
    model = load_rtdetr(RTDETR_XL_WEIGHTS)
    model.to(args.device)

    results = []
    t0 = time.time()

    for i, vinfo in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {vinfo['stem']} ({vinfo['scene_layout']}, GT={vinfo['gt_type']})")
        try:
            result = process_video(vinfo, model, args.device)
            result["gt_type"] = vinfo["gt_type"]
            result["scene_layout"] = vinfo["scene_layout"]
            result["path"] = vinfo["path"]
            results.append(result)

            ok = result["pred_type"] == vinfo["gt_type"]
            print(
                f"  -> raw={result['raw_type']}, final={result['pred_type']}, GT={vinfo['gt_type']} "
                f"[{'OK' if ok else 'MISS'}]  (dets={result['n_dets']}, tracks={result['n_tracks']}, coll={result['n_coll']})"
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "stem": vinfo["stem"], "raw_type": "single", "pred_type": "single",
                "gt_type": vinfo["gt_type"], "scene_layout": vinfo["scene_layout"],
                "path": vinfo["path"], "n_dets": 0, "n_tracks": 0, "n_coll": 0, "n_anom": 0,
            })

    elapsed = time.time() - t0
    print(f"\nProcessed {len(results)} videos in {elapsed:.1f}s ({elapsed / max(len(results), 1):.1f}s/video)")

    with open(OUTPUT_DIR / "intersection_results.json", "w") as f:
        json.dump(results, f, indent=2)

    total = len(results)
    correct_new = sum(1 for r in results if r["pred_type"] == r["gt_type"])
    c_new = correct_new / total if total else 0.0

    sub5_path = _DATA / "test_metadata" / "submission5_qwen35_spatial.csv"
    with open(sub5_path, newline="") as f:
        sub5 = {r["path"]: r for r in csv.DictReader(f)}

    correct_sub5 = sum(
        1 for r in results
        if sub5.get(r["path"], {}).get("type", "").strip().lower() == r["gt_type"]
    )
    c_sub5 = correct_sub5 / total if total else 0.0

    print(f"\n{'='*70}")
    print(f"C on intersection (n={total})")
    print(f"  Sub5 (L + ByteTrack + rerank in pipeline): {correct_sub5}/{total} = {c_sub5:.4f}")
    print(f"  RT-DETR-XL + ByteTrack + same rerank:       {correct_new}/{total} = {c_new:.4f}")
    print(f"  Delta: {c_new - c_sub5:+.4f}")

    by_scene = defaultdict(list)
    for r in results:
        by_scene[r["scene_layout"]].append(r)

    print("\nPer scene_layout:")
    for scene, rs in sorted(by_scene.items()):
        n = len(rs)
        new_c = sum(1 for r in rs if r["pred_type"] == r["gt_type"])
        old_c = sum(1 for r in rs if sub5.get(r["path"], {}).get("type", "").strip().lower() == r["gt_type"])
        print(f"  {scene:35s}  n={n:3d}  Sub5={old_c}/{n}={old_c/n:.3f}  XL={new_c}/{n}={new_c/n:.3f}  d={new_c/n-old_c/n:+.3f}")

    comp_path = OUTPUT_DIR / "comparison.csv"
    with open(comp_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "stem", "scene_layout", "gt_type", "sub5_type", "raw_type", "final_type", "sub5_ok", "xl_ok"])
        for r in results:
            s5t = sub5.get(r["path"], {}).get("type", "").strip().lower()
            w.writerow([
                r["path"], r["stem"], r["scene_layout"], r["gt_type"], s5t,
                r["raw_type"], r["pred_type"],
                1 if s5t == r["gt_type"] else 0,
                1 if r["pred_type"] == r["gt_type"] else 0,
            ])

    print(f"\nWrote {OUTPUT_DIR}/intersection_results.json and comparison.csv")


if __name__ == "__main__":
    main()
