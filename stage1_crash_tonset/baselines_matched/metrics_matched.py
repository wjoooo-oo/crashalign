#!/usr/bin/env python3
"""Val-selected threshold + test metrics, matching the h_only protocol.

阈值在 val 上按 balanced accuracy 选取，冻结后用于 test。
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_balanced_threshold(y_true, y_score, step: float = 0.01) -> tuple[float, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    lo, hi = float(y_score.min()), float(y_score.max())
    if hi - lo < 1e-12:
        return 0.5, float(balanced_accuracy_score(y_true, np.zeros_like(y_true)))
    taus = np.linspace(lo, hi, max(3, int(round(1.0 / step)) + 1))
    best_tau, best_bal = float(taus[0]), -1.0
    for tau in taus:
        pred = (y_score >= tau).astype(int)
        bal = float(balanced_accuracy_score(y_true, pred))
        if bal > best_bal:
            best_bal = bal
            best_tau = float(tau)
    return best_tau, best_bal


def crash_metrics(y_true, y_score, thr: float) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= thr).astype(int)
    return {
        "threshold": float(thr),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "crash_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "noncrash_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None,
        "auprc": float(average_precision_score(y_true, y_score)) if y_true.sum() > 0 else None,
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
        "n": int(len(y_true)),
        "n_crash": int((y_true == 1).sum()),
        "n_non": int((y_true == 0).sum()),
    }


def write_scores(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["clip_id", "video_id", "role", "is_crash", "score"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})


def save_run(out_dir: Path, *, method: str, val_rows: list[dict], test_rows: list[dict], extra: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    yv = [r["is_crash"] for r in val_rows]
    sv = [r["score"] for r in val_rows]
    thr, val_ba = best_balanced_threshold(yv, sv)
    val_m = crash_metrics(yv, sv, thr)
    yt = [r["is_crash"] for r in test_rows]
    st = [r["score"] for r in test_rows]
    test_m = crash_metrics(yt, st, thr)
    payload = {
        "protocol": "ACCIDENT matched fixed-L pos/neg",
        "method": method,
        "threshold_source": "val balanced accuracy",
        "val": val_m,
        "test": test_m,
        "extra": extra or {},
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2))
    write_scores(out_dir / "scores_val.csv", val_rows)
    write_scores(out_dir / "scores_test.csv", test_rows)
    md = (
        f"# {method}\n\n"
        f"Threshold from val BA: **{thr:.4f}** (val BA {val_ba:.4f})\n\n"
        f"| Split | Bal Acc | Crash Rec | Non-crash Rec | Prec | F1 | AUROC | AUPRC |\n"
        f"|---|---:|---:|---:|---:|---:|---:|---:|\n"
        f"| val | {val_m['balanced_accuracy']:.4f} | {val_m['crash_recall']:.4f} | "
        f"{val_m['noncrash_recall']:.4f} | {val_m['precision']:.4f} | {val_m['f1']:.4f} | "
        f"{val_m['auroc']:.4f} | {val_m['auprc']:.4f} |\n"
        f"| **test** | **{test_m['balanced_accuracy']:.4f}** | {test_m['crash_recall']:.4f} | "
        f"{test_m['noncrash_recall']:.4f} | {test_m['precision']:.4f} | {test_m['f1']:.4f} | "
        f"{test_m['auroc']:.4f} | {test_m['auprc']:.4f} |\n"
    )
    (out_dir / "report.md").write_text(md)
    print(md)
    return payload
