#!/usr/bin/env python3
"""
0625_Evaluate sub5 (or any submission) TSC on the LoRA test split.

Uses ACCIDENT paper metrics; default spatial mode matches competition (isotropic sigma=0.1).
使用 ACCIDENT 论文指标；默认空间模式与竞赛一致（各向同性 sigma=0.1）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from crashalign_paths import accident_root  # noqa: E402
from accident_metrics import evaluate_submission_csv

HERE = Path(__file__).resolve().parent
_DATA = accident_root()
GT_CSV = _DATA / "ACCIDENT_dataset" / "ground_truth_full.csv"
SUB5_CSV = _DATA / "test_metadata" / "submission5_qwen35_spatial.csv"
TEST_JSONL = HERE / "data" / "test.jsonl"


def load_test_paths() -> list[str]:
    paths = []
    for line in TEST_JSONL.read_text(encoding="utf-8").splitlines():
        if line.strip():
            paths.append(json.loads(line)["path"])
    return paths


def main():
    parser = argparse.ArgumentParser(description="TSC eval on LoRA test split")
    parser.add_argument("--submission", type=str, default=str(SUB5_CSV))
    parser.add_argument("--gt", type=str, default=str(GT_CSV))
    parser.add_argument("--sigma_t", type=float, default=2.0)
    parser.add_argument("--sigma_space", type=float, default=0.1)
    parser.add_argument("--anisotropic", action="store_true", help="Use paper Eq. (3)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    paths = load_test_paths()
    scores = evaluate_submission_csv(
        args.submission,
        args.gt,
        paths=paths,
        sigma_t=args.sigma_t,
        sigma_space=args.sigma_space,
        anisotropic=args.anisotropic,
    )
    text = json.dumps(scores, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
