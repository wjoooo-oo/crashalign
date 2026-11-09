#!/usr/bin/env python3
"""Sanity-check ACCIDENT matched split, cache, clip catalog, and h_only numbers."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from clip_io import (
    CACHE_DIR,
    CATALOG_DIR,
    DATA_DIR,
    ROOT,
    build_split_catalog,
    ensure_catalogs,
    save_catalog,
    validate_specs,
)

EXPECTED = {"train": 514, "val": 75, "test": 132}
H_ONLY = {
    "balanced_accuracy": 0.6022727272727273,
    "crash_recall": 0.75,
    "noncrash_recall": 0.45454545454545453,
    "f1": 0.6534653465346535,
}


def count_jsonl(path: Path) -> int:
    return sum(1 for l in path.read_text().splitlines() if l.strip())


def main() -> None:
    print("=== split counts ===")
    for split, n in EXPECTED.items():
        got = count_jsonl(DATA_DIR / f"{split}_matched.jsonl")
        cache_n = len(list((CACHE_DIR / split).glob("*.npz")))
        print(f"  {split}: matched={got} (expect {n}), cache_npz={cache_n}")
        if got != n:
            raise SystemExit(f"{split} matched count {got} != {n}")

    meta = json.loads((DATA_DIR / "split_meta.json").read_text())
    print("  split_meta.final_matched:", meta.get("final_matched"))
    print("  cache extra in train (excluded video) is OK if present")

    print("=== clip catalog ===")
    catalogs = {}
    for split in EXPECTED:
        specs = build_split_catalog(split, train_mode=False)
        save_catalog(split, specs)
        catalogs[split] = specs
        errs = validate_specs(specs)
        n_pos = sum(s.is_crash for s in specs)
        n_neg = len(specs) - n_pos
        durs = [s.t_end - s.t_start for s in specs]
        print(
            f"  {split}: {len(specs)} clips ({n_pos} pos / {n_neg} neg), "
            f"dur mean={sum(durs)/len(durs):.2f}s, catalog -> {CATALOG_DIR / (split + '.jsonl')}"
        )
        if errs:
            print("  FIRST ERRORS:")
            for e in errs[:8]:
                print("   ", e)
            raise SystemExit(f"{split} catalog has {len(errs)} protocol errors")
        if n_pos != EXPECTED[split] or n_neg != EXPECTED[split]:
            raise SystemExit(f"{split} pos/neg imbalance {n_pos}/{n_neg}")

    print("=== re-eval h_only ===")
    py = sys.executable
    out = Path("/tmp/h_only_recheck")
    cmd = [
        py,
        "-u",
        str(ROOT / "0726_eval_accident_matched.py"),
        "--checkpoints",
        str(ROOT / "checkpoints/accident_matched/h_only/best.pt"),
        "--matched_jsonl",
        str(DATA_DIR / "test_matched.jsonl"),
        "--cache_dir",
        str(CACHE_DIR / "test"),
        "--fixed_len",
        "8",
        "--seed",
        "42",
        "--output_dir",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    report = json.loads((out / "metrics.json").read_text())
    crash = report["runs"][0]["crash"]
    for k, exp in H_ONLY.items():
        got = crash[k]
        if abs(got - exp) > 1e-6:
            raise SystemExit(f"h_only {k} {got} != {exp}")
        print(f"  {k}: {got:.4f} OK")
    print("Protocol + h_only check passed.")


if __name__ == "__main__":
    main()
