#!/usr/bin/env python3
"""Build deterministic matched clip catalogs (no model eval).

The split comes from clip_io.DATA_DIR (env MATCHED_DATA_DIR); every matched video must
yield exactly one positive and one negative clip.
划分来自 clip_io.DATA_DIR（环境变量 MATCHED_DATA_DIR）；每个 matched 视频须恰好生成一正一负。
"""

from __future__ import annotations

from clip_io import CATALOG_DIR, DATA_DIR, build_split_catalog, load_jsonl, save_catalog, validate_specs


def main():
    for split in ("train", "val", "test"):
        n_pairs = len(load_jsonl(DATA_DIR / f"{split}_matched.jsonl"))
        specs = build_split_catalog(split, train_mode=False)
        errs = validate_specs(specs)
        if errs:
            raise SystemExit(f"{split}: {len(errs)} errors, e.g. {errs[0]}")
        n_pos = sum(s.is_crash for s in specs)
        path = save_catalog(split, specs)
        print(f"{split}: {len(specs)} clips ({n_pos} pos / {len(specs)-n_pos} neg) -> {path}")
        if n_pos != n_pairs:
            raise SystemExit(f"{split} expected {n_pairs} pairs, got {n_pos}")
    print(f"Catalogs written under {CATALOG_DIR}")


if __name__ == "__main__":
    main()
