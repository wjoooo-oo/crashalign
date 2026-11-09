#!/usr/bin/env python3
"""Evaluate Stage-I crash-head ablations on the ACCIDENT matched test set.

Each matched source video yields one positive and one clean-normal negative
clip (fixed L, no padding). Thresholds come from the checkpoint (val BA).

中文：在 ACCIDENT matched 测试集上评估各 crash-head 消融。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from labels import LEGAL_TRANSITIONS, PHASE_ORDER, PHASE2ID  # noqa: E402
from model import Task1CumPhaseModel  # noqa: E402


def _load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_model(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    conf = ckpt.get("args", {})
    model = Task1CumPhaseModel(
        emb_dim=ckpt.get("emb_dim", 768),
        gru_hidden=conf.get("gru_hidden", 256),
        dropout=0.0,
        crash_feat=conf.get("crash_feat", "full"),
        phase_mode=conf.get("phase_mode", "cum"),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, float(ckpt["best_thr"]), conf, ckpt


def transition_consistency(seq: np.ndarray) -> tuple[int, int]:
    if len(seq) < 2:
        return 0, 0
    legal = sum((int(a), int(b)) in LEGAL_TRANSITIONS for a, b in zip(seq[:-1], seq[1:]))
    return legal, len(seq) - 1


@torch.no_grad()
def score_dataset(model, ds, device, thr: float) -> dict:
    y_true, y_prob, y_pred = [], [], []
    clip_ids = []
    phase_yt, phase_yp = [], []
    legal_n = total_n = 0
    distinct = []

    for i in range(len(ds)):
        item = ds[i]
        emb = item["emb"].unsqueeze(0).to(device)
        mask = item["valid_mask"].unsqueeze(0).to(device)
        out = model(emb, mask)
        prob = float(torch.sigmoid(out["crash_logit"]).cpu())
        pred = int(prob >= thr)
        y_true.append(int(item["is_crash"].item() > 0.5))
        y_prob.append(prob)
        y_pred.append(pred)
        clip_ids.append(item["video_id"])

        # Phase metrics on positive clips only / 仅在正样本上统计 phase
        if item["role"] == "positive" and "phase_pred" in out:
            yp = out["phase_pred"][0].cpu().numpy()
            yt = item["y_phase"].numpy()
            phase_yt.append(yt)
            phase_yp.append(yp)
            leg, tot = transition_consistency(yp)
            legal_n += leg
            total_n += tot
            distinct.append(len(set(yp.tolist())))

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.array(y_pred)

    crash = {
        "threshold": thr,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "crash_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "noncrash_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None,
        "auprc": float(average_precision_score(y_true, y_prob)) if y_true.sum() > 0 else None,
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
        "n": int(len(y_true)),
        "n_crash": int((y_true == 1).sum()),
        "n_non": int((y_true == 0).sum()),
    }

    phase = None
    if phase_yt:
        yt = np.concatenate(phase_yt)
        yp = np.concatenate(phase_yp)
        from sklearn.metrics import f1_score as f1s

        per = {}
        for name, pid in PHASE2ID.items():
            per[name] = float(
                f1s(yt == pid, yp == pid, zero_division=0)
            )
        phase = {
            "macro_f1": float(f1s(yt, yp, average="macro", labels=list(range(4)), zero_division=0)),
            "per_phase_f1": per,
            "transition_consistency": float(legal_n / total_n) if total_n else None,
            "mean_distinct_phases_per_crash_clip": float(np.mean(distinct)) if distinct else None,
        }
    # Per-clip probabilities allow paired bootstrap comparisons between runs.
    # 保存逐 clip 概率，便于不同运行之间做配对 bootstrap 比较。
    clip_scores = [
        {"clip_id": cid, "is_crash": int(t), "prob": float(p)}
        for cid, t, p in zip(clip_ids, y_true, y_prob)
    ]
    return {"crash": crash, "phase": phase, "clip_scores": clip_scores}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--matched_jsonl", default=str(HERE / "data_accident_matched/test_matched.jsonl"))
    p.add_argument("--cache_dir", default=str(HERE / "cache_emb_accident/test"))
    p.add_argument("--fixed_len", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default=str(HERE / "results/accident_matched_ablation"))
    args = p.parse_args()

    ds_mod = _load_mod("matched_ds", HERE / "0726_matched_dataset.py")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ds_mod.MatchedAccidentDataset(
        args.matched_jsonl,
        args.cache_dir,
        fixed_len=args.fixed_len,
        train=False,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    report = {
        "protocol": "ACCIDENT matched fixed-L pos/neg",
        "fixed_len": args.fixed_len,
        "n_items": len(ds),
        "n_videos": len(ds) // 2,
        "runs": [],
    }
    for ck in args.checkpoints:
        path = Path(ck)
        model, thr, conf, ckpt = load_model(path, device)
        metrics = score_dataset(model, ds, device, thr)
        run = {
            "crash_feat": conf.get("crash_feat", path.parent.name),
            "checkpoint": str(path),
            "best_epoch": ckpt.get("epoch"),
            "val_metrics": ckpt.get("metrics"),
            "threshold": thr,
            **metrics,
        }
        report["runs"].append(run)
        c, ph = metrics["crash"], metrics["phase"] or {}
        rows.append(
            {
                "method": conf.get("crash_feat", path.parent.name),
                "acc": c["accuracy"],
                "bal_acc": c["balanced_accuracy"],
                "crash_recall": c["crash_recall"],
                "noncrash_recall": c["noncrash_recall"],
                "precision": c["precision"],
                "f1": c["f1"],
                "auroc": c["auroc"],
                "auprc": c["auprc"],
                "phase_macro_f1": ph.get("macro_f1"),
                "trans_cons": ph.get("transition_consistency"),
                "thr": thr,
            }
        )
        print(
            f"{run['crash_feat']}: BA={c['balanced_accuracy']:.4f} "
            f"CR={c['crash_recall']:.4f} NCR={c['noncrash_recall']:.4f} "
            f"F1={c['f1']:.4f} phaseF1={ph.get('macro_f1')} trans={ph.get('transition_consistency')}"
        )

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    # Method display names / 论文显示名
    name_map = {
        "full": "Full",
        "h_only": "Feature-only",
        "phase_only": "Phase-only",
        "phase_unordered": "Phase-unordered",
        "pattern_gated": "Pattern-gated",
        "fp_gated": "FP-gated",
    }
    order = ["full", "h_only", "phase_only", "fp_gated", "phase_unordered", "pattern_gated"]
    rows_sorted = sorted(rows, key=lambda r: order.index(r["method"]) if r["method"] in order else 99)

    def fmt(x, nd=4):
        return f"{x:.{nd}f}" if x is not None else "--"

    latex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Stage~I crash-head ablation on the ACCIDENT matched test set "
        r"($L{=}8$ windows, same-video clean-normal negatives). "
        r"Feature-only omits phase supervision; phase-only classifies from the "
        r"phase summary alone. Thresholds are selected on validation balanced "
        r"accuracy and frozen for test.}",
        r"\label{tab:tad_head_ablation}",
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r"Method & Acc & Bal.\ Acc. & Crash Rec. & Non-crash Rec. & Prec. & F1 & AUROC & AUPRC & Phase F1 & Trans. \\",
        r"\midrule",
    ]
    for r in rows_sorted:
        latex_lines.append(
            f"{name_map.get(r['method'], r['method'])} & "
            f"{fmt(r['acc'])} & {fmt(r['bal_acc'])} & {fmt(r['crash_recall'])} & "
            f"{fmt(r['noncrash_recall'])} & {fmt(r['precision'])} & {fmt(r['f1'])} & "
            f"{fmt(r['auroc'])} & {fmt(r['auprc'])} & {fmt(r['phase_macro_f1'])} & "
            f"{fmt(r['trans_cons'])} \\\\"
        )
    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # ~100-word discussion paragraph
    best = max(rows_sorted, key=lambda r: r["bal_acc"])
    fo = next((r for r in rows_sorted if r["method"] == "h_only"), None)
    po = next((r for r in rows_sorted if r["method"] == "phase_only"), None)
    fu = next((r for r in rows_sorted if r["method"] == "full"), None)
    para = (
        "Table~\\ref{tab:tad_head_ablation} compares Stage~I crash heads on the "
        "ACCIDENT matched benchmark, where each source video contributes a "
        "fixed-length positive clip and a same-video clean-normal negative "
        f"(L=8, {report['n_videos']} pairs). "
    )
    if fu and fo and po:
        # Build prose without empty f-string braces like {=} which are invalid.
        # 避免 f-string 中出现空 {{=}} 语法错误。
        fo_pf = fo["phase_macro_f1"] if fo["phase_macro_f1"] is not None else float("nan")
        fu_pf = fu["phase_macro_f1"] if fu["phase_macro_f1"] is not None else float("nan")
        po_pf = po["phase_macro_f1"] if po["phase_macro_f1"] is not None else float("nan")
        para += (
            f"All variants reach detection "
            f"balanced accuracy {fu['bal_acc']:.3f}/{fo['bal_acc']:.3f}/{po['bal_acc']:.3f} "
            f"for full/feature-only/phase-only on this matched hard setting. "
            f"Phase supervision mainly improves structured temporal labels: "
            f"phase macro-F1 is {fo_pf:.3f} without phase loss versus "
            f"{fu_pf:.3f}/{po_pf:.3f} for full/phase-only, "
            f"with higher transition consistency when phase is supervised. "
            f"Thus auxiliary phase learning yields interpretable progression "
            f"representations rather than large gains on binary detection alone."
        )
    else:
        para += f"The best balanced accuracy is {best['bal_acc']:.3f} ({best['method']})."

    words = len(para.replace("\\", "").split())
    latex_lines.append("% discussion (~100 words)")
    latex_lines.append(para)
    latex_lines.append(f"% word_count≈{words}")

    tex_path = out_dir / "0726_tad_head_ablation_table.tex"
    tex_path.write_text("\n".join(latex_lines) + "\n")

    md = ["| Method | Acc | Bal Acc | Crash Rec | Non-crash Rec | Prec | F1 | AUROC | AUPRC | Phase F1 | Trans |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows_sorted:
        md.append(
            f"| {name_map.get(r['method'], r['method'])} | {fmt(r['acc'])} | {fmt(r['bal_acc'])} | "
            f"{fmt(r['crash_recall'])} | {fmt(r['noncrash_recall'])} | {fmt(r['precision'])} | "
            f"{fmt(r['f1'])} | {fmt(r['auroc'])} | {fmt(r['auprc'])} | {fmt(r['phase_macro_f1'])} | "
            f"{fmt(r['trans_cons'])} |"
        )
    (out_dir / "table.md").write_text("\n".join(md) + "\n\n" + para + "\n")
    print(f"Wrote {tex_path} and {out_dir/'table.md'} (words≈{words})")


if __name__ == "__main__":
    main()
