#!/usr/bin/env python3
"""
Evaluate Qwen vehicle class predictions against RT-DETR pseudo-GT.

Usage:
  python 0609crashexplanation/eval_vehicle_class.py \
    --results_csv results/frame_ablation_200gt/v6_pre13_limit200.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from crashalign_paths import accident_root  # noqa: E402

ROOT = accident_root()

from supportive_eval import score_vehicle_classes

GT_CSV = ROOT / "test_200_GT" / "ground_truth_200.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_csv", required=True, help="Qwen explanation results CSV")
    parser.add_argument("--gt_csv", default=str(GT_CSV), help="Ground truth CSV with vehicle_classes column")
    args = parser.parse_args()

    results = pd.read_csv(args.results_csv)
    gt = pd.read_csv(args.gt_csv)
    
    if "vehicle_classes" not in gt.columns:
        print("Error: vehicle_classes column not found in GT CSV. Run build_vehicle_class_gt.py first.")
        return
    
    gt_map = gt.set_index("path").to_dict(orient="index")

    scores = []
    for _, row in results.iterrows():
        path = row["path"]
        gt_row = gt_map.get(path, {})
        gt_classes_str = gt_row.get("vehicle_classes", "")
        gt_classes = [c.strip() for c in str(gt_classes_str).split(",") if c.strip()] if gt_classes_str else []
        
        pred_classes_str = row.get("vehicle_classes_pred", "[]")
        try:
            pred_classes = json.loads(pred_classes_str) if isinstance(pred_classes_str, str) else pred_classes_str
        except json.JSONDecodeError:
            pred_classes = []
        
        sc = score_vehicle_classes(pred_classes, gt_classes)
        scores.append({
            "path": path,
            "gt_classes": ",".join(gt_classes),
            "pred_classes": ",".join(pred_classes),
            "recall": sc["recall"],
            "precision": sc["precision"],
            "f1": sc["f1"],
        })

    df = pd.DataFrame(scores)
    
    valid = df[df["recall"].notna()]
    print(f"\nVehicle Class Evaluation (n={len(valid)} with GT)")
    print("=" * 60)
    print(f"  Recall:    {valid['recall'].mean():.3f} (GT classes mentioned by Qwen)")
    print(f"  Precision: {valid['precision'].mean():.3f} (Qwen classes in GT)")
    print(f"  F1:        {valid['f1'].mean():.3f}")
    print()
    
    # Per-class breakdown
    from collections import Counter
    gt_counter = Counter()
    pred_counter = Counter()
    hit_counter = Counter()
    
    for _, row in df.iterrows():
        for c in row["gt_classes"].split(","):
            if c.strip():
                gt_counter[c.strip()] += 1
        for c in row["pred_classes"].split(","):
            if c.strip():
                pred_counter[c.strip()] += 1
        gt_set = set(c.strip() for c in row["gt_classes"].split(",") if c.strip())
        pred_set = set(c.strip() for c in row["pred_classes"].split(",") if c.strip())
        for c in gt_set & pred_set:
            hit_counter[c] += 1
    
    print("Per-class breakdown:")
    print(f"  {'class':<12} {'GT':>6} {'Pred':>6} {'Hit':>6} {'Recall':>8}")
    print("-" * 44)
    for cls in sorted(set(gt_counter.keys()) | set(pred_counter.keys())):
        g = gt_counter[cls]
        p = pred_counter[cls]
        h = hit_counter[cls]
        r = h / g if g > 0 else 0
        print(f"  {cls:<12} {g:>6} {p:>6} {h:>6} {r:>8.3f}")
    
    out_path = Path(args.results_csv).with_suffix(".vehicle_class_eval.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved per-video scores to {out_path}")


if __name__ == "__main__":
    main()
