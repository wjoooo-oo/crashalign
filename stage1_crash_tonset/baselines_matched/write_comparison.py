#!/usr/bin/env python3
"""Assemble the ACCIDENT matched TAD comparison table (ours = h_only only)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/accident_matched_baselines"
H_ONLY_METRICS = ROOT / "results/accident_matched_ablation/metrics.json"

METHODS = [
    ("h_only", "Ours (h_only / Feature-only)", True),
    ("cadp_lstm", "CADP / Accident LSTM", False),
    ("stacked_autoencoder", "Stacked Autoencoder", False),
    ("tcff_gan", "TCFF-GAN / SO-TAD", False),
    ("mdts_adnet", "MDTS-ADNet", False),
    ("qwen35_27b_zeroshot", "Qwen3.5-27B zero-shot", False),
    ("videomae_accident", "VideoMAE (ACCIDENT clips)", False),
]


def fmt(x, nd=4):
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) and x is not None else "--"


def load_h_only() -> dict | None:
    if not H_ONLY_METRICS.is_file():
        return None
    report = json.loads(H_ONLY_METRICS.read_text())
    for run in report.get("runs", []):
        if run.get("crash_feat") == "h_only":
            return run["crash"]
    return None


def load_baseline(name: str) -> dict | None:
    p = OUT / name / "metrics.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("test")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, label, is_ours in METHODS:
        m = load_h_only() if is_ours else load_baseline(key)
        rows.append((label, m, "ok" if m else "pending"))

    md = [
        "# ACCIDENT matched TAD comparison",
        "",
        "Protocol: same-video fixed-L=8 positive / clean-normal negative, test 132 pairs (264 clips).",
        "Thresholds selected on val balanced accuracy (except h_only, already frozen from its checkpoint).",
        "Ours reports **h_only only**. SOTAD official-test numbers are a different protocol and are not comparable.",
        "",
        "| Method | Bal Acc | Crash Rec | Non-crash Rec | Prec | F1 | AUROC | AUPRC | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, m, status in rows:
        if m is None:
            md.append(f"| {label} | -- | -- | -- | -- | -- | -- | -- | {status} |")
            continue
        md.append(
            f"| {label} | {fmt(m.get('balanced_accuracy'))} | {fmt(m.get('crash_recall'))} | "
            f"{fmt(m.get('noncrash_recall'))} | {fmt(m.get('precision'))} | {fmt(m.get('f1'))} | "
            f"{fmt(m.get('auroc'))} | {fmt(m.get('auprc'))} | {status} |"
        )
    text = "\n".join(md) + "\n"
    (OUT / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
