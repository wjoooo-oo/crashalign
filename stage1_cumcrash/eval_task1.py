#!/usr/bin/env python3
"""Evaluate Task1 cumulative crash pattern on cached test embeddings.

Reports:
  - crash/non metrics at several thresholds
  - pattern quality on crash: MAE of transition point vs t_gt (aux only)
  - non: fraction with spurious rising q_pre / q_post
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import CumCrashModel  # noqa: E402


def load_split(cache_dir: Path):
    files = sorted(cache_dir.glob("*.npz"))
    return files


@torch.no_grad()
def score_file(model, path: Path, device):
    with np.load(path) as z:
        emb = torch.from_numpy(z["emb"].astype(np.float32)).unsqueeze(0).to(device)
        times = z["times"].astype(np.float32)
        y_pre = z["y_pre"].astype(np.float32)
        y_post = z["y_post"].astype(np.float32)
        is_crash = int(z["is_crash"])
        t_gt = float(z["t_gt"])
        vid = str(z["video_id"])
    mask = torch.ones(1, emb.size(1), dtype=torch.bool, device=device)
    out = model(emb, mask)
    q_pre = out["q_pre"][0].cpu().numpy()
    q_post = out["q_post"][0].cpu().numpy()
    crash_prob = float(torch.sigmoid(out["crash_logit"][0]).cpu())
    # transition ≈ where pre high and post starts rising
    gap = np.abs(q_pre - 0.5) + np.abs(q_post - 0.5)
    # prefer time where q_pre crossed 0.5 before q_post
    cross = np.where((q_pre >= 0.5) & (q_post < 0.5))[0]
    if len(cross):
        t_trans = float(times[int(cross[0])])
    else:
        t_trans = float(times[int(np.argmin(np.abs(q_pre - q_post)))]) if len(times) else None
    return {
        "video_id": vid,
        "is_crash": is_crash,
        "t_gt": t_gt,
        "crash_prob": crash_prob,
        "max_q_pre": float(q_pre.max()),
        "max_q_post": float(q_post.max()),
        "last_q_pre": float(q_pre[-1]),
        "last_q_post": float(q_post[-1]),
        "t_transition": t_trans,
        "pre_bce": float(
            -np.mean(y_pre * np.log(q_pre + 1e-6) + (1 - y_pre) * np.log(1 - q_pre + 1e-6))
        ),
        "post_bce": float(
            -np.mean(y_post * np.log(q_post + 1e-6) + (1 - y_post) * np.log(1 - q_post + 1e-6))
        ),
    }


def det_at(rows, thr, key="crash_prob"):
    crash = [r for r in rows if r["is_crash"] == 1]
    non = [r for r in rows if r["is_crash"] == 0]
    rec = np.mean([r[key] >= thr for r in crash]) if crash else None
    spec = np.mean([r[key] < thr for r in non]) if non else None
    fa = int(sum(r[key] >= thr for r in non))
    miss = int(sum(r[key] < thr for r in crash))
    return {
        "threshold": thr,
        "recall": float(rec) if rec is not None else None,
        "specificity": float(spec) if spec is not None else None,
        "bal_acc": float(0.5 * (rec + spec)) if rec is not None and spec is not None else None,
        "n_false_alarm": fa,
        "n_miss": miss,
        "n_crash": len(crash),
        "n_non": len(non),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=str(HERE / "checkpoints/cumcrash_v1/best.pt"))
    p.add_argument("--cache_dir", default=str(HERE / "cache_emb"))
    p.add_argument("--split", default="test", choices=("val", "test"))
    p.add_argument("--output_dir", default=None)
    p.add_argument("--gru_hidden", type=int, default=256)
    args = p.parse_args()

    out_dir = Path(args.output_dir or (HERE / "results" / f"eval_{args.split}"))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = load_split(Path(args.cache_dir) / args.split)
    if not files:
        raise SystemExit(f"No embeddings in {Path(args.cache_dir) / args.split}")

    with np.load(files[0]) as z:
        d = z["emb"].shape[-1]
    ckpt = torch.load(args.checkpoint, map_location=device)
    hidden = ckpt.get("args", {}).get("gru_hidden", args.gru_hidden)
    model = CumCrashModel(emb_dim=d, gru_hidden=hidden).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    rows = []
    for f in tqdm(files, desc=f"eval-{args.split}"):
        rows.append(score_file(model, f, device))

    with (out_dir / "predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    crash_rows = [r for r in rows if r["is_crash"] == 1]
    non_rows = [r for r in rows if r["is_crash"] == 0]
    trans_err = [
        abs(r["t_transition"] - r["t_gt"])
        for r in crash_rows
        if r["t_gt"] >= 0 and r["t_transition"] is not None
    ]
    summary = {
        "split": args.split,
        "checkpoint": args.checkpoint,
        "n": len(rows),
        "n_crash": len(crash_rows),
        "n_non": len(non_rows),
        "det_crash_prob": {
            "thr_0.5": det_at(rows, 0.5),
            "thr_0.35": det_at(rows, 0.35),
            "thr_0.65": det_at(rows, 0.65),
        },
        "det_max_q_pre": {
            "thr_0.5": det_at(rows, 0.5, key="max_q_pre"),
            "thr_0.35": det_at(rows, 0.35, key="max_q_pre"),
        },
        "pattern_crash": {
            "transition_mae_sec": float(np.mean(trans_err)) if trans_err else None,
            "transition_median_sec": float(np.median(trans_err)) if trans_err else None,
            "mean_max_q_pre": float(np.mean([r["max_q_pre"] for r in crash_rows])) if crash_rows else None,
            "mean_max_q_post": float(np.mean([r["max_q_post"] for r in crash_rows])) if crash_rows else None,
        },
        "pattern_non": {
            "frac_max_q_pre_ge_0.5": float(np.mean([r["max_q_pre"] >= 0.5 for r in non_rows])) if non_rows else None,
            "frac_max_q_post_ge_0.5": float(np.mean([r["max_q_post"] >= 0.5 for r in non_rows])) if non_rows else None,
            "mean_max_q_pre": float(np.mean([r["max_q_pre"] for r in non_rows])) if non_rows else None,
            "mean_max_q_post": float(np.mean([r["max_q_post"] for r in non_rows])) if non_rows else None,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
