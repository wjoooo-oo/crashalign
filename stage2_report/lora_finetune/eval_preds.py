#!/usr/bin/env python3
"""
0625_Evaluate LoRA crash-report predictions on test jsonl.

Reports fixed-field accuracy, context mean, participant F1, and reasoning coverage.
报告固定字段准确率、context 均值、participants F1 与 reasoning 覆盖率。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from supportive_eval import score_vehicle_classes  # noqa: E402

FIXED_FIELDS = ["scene", "weather", "lighting", "collision_type"]
REASON_FIELDS = ["observation", "explanation", "summary"]


def _non_empty(val) -> bool:
    if val is None:
        return False
    if isinstance(val, list):
        return any(str(x).strip() for x in val)
    return bool(str(val).strip())


def _load_source_map(test_jsonl: Path | None) -> dict[str, str]:
    if test_jsonl is None or not test_jsonl.is_file():
        return {}
    out: dict[str, str] = {}
    for line in test_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[str(rec["path"])] = str(rec.get("source", "unknown"))
    return out


def _summarize_rows(rows: list[dict]) -> dict:
    n = len(rows)
    hits = {f: 0 for f in FIXED_FIELDS}
    n_parse = 0
    context_hits = 0
    context_n = 0
    reason_ok = 0
    part_recalls, part_precs, part_f1s = [], [], []

    for row in rows:
        pred = row.get("pred") or {}
        tgt = row.get("target") or {}
        if pred:
            n_parse += 1

        for f in ["scene", "weather", "lighting"]:
            ok = pred.get(f) is not None and str(pred.get(f)) == str(tgt.get(f))
            context_n += 1
            context_hits += int(ok)

        for f in FIXED_FIELDS:
            if pred.get(f) is not None and str(pred.get(f)) == str(tgt.get(f)):
                hits[f] += 1

        if all(_non_empty(pred.get(f)) for f in REASON_FIELDS):
            reason_ok += 1

        gt_parts = [str(x).lower() for x in (tgt.get("participants") or [])]
        pred_parts = [str(x).lower() for x in (pred.get("participants") or [])]
        sc = score_vehicle_classes(pred_parts, gt_parts)
        if sc["f1"] is not None:
            part_recalls.append(sc["recall"])
            part_precs.append(sc["precision"])
            part_f1s.append(sc["f1"])

    def _rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    return {
        "n": n,
        "parse_rate": _rate(n_parse, n),
        "scene_acc": _rate(hits["scene"], n),
        "weather_acc": _rate(hits["weather"], n),
        "lighting_acc": _rate(hits["lighting"], n),
        "collision_type_acc": _rate(hits["collision_type"], n),
        "context_mean": _rate(context_hits, context_n),
        "participants_recall": round(sum(part_recalls) / len(part_recalls), 4) if part_recalls else None,
        "participants_precision": round(sum(part_precs) / len(part_precs), 4) if part_precs else None,
        "participants_f1": round(sum(part_f1s) / len(part_f1s), 4) if part_f1s else None,
        "reasoning_coverage": _rate(reason_ok, n),
    }


def eval_preds(
    preds_path: Path,
    *,
    test_jsonl: Path | None = None,
    by_source: bool = True,
) -> dict:
    rows = [
        json.loads(line)
        for line in preds_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_map = _load_source_map(test_jsonl)
    for row in rows:
        if "source" not in row:
            row["source"] = source_map.get(str(row.get("path", "")), "unknown")

    by_src: dict[str, list] = {}
    for row in rows:
        by_src.setdefault(str(row.get("source", "unknown")), []).append(row)

    summary = _summarize_rows(rows)
    if by_source and by_src:
        summary["by_source"] = {src: _summarize_rows(src_rows) for src, src_rows in sorted(by_src.items())}
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate preds jsonl")
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--test_jsonl", type=str, default=str(HERE / "data" / "test.jsonl"))
    parser.add_argument("--out", type=str, default=None, help="Optional JSON summary path")
    args = parser.parse_args()

    summary = eval_preds(
        Path(args.preds),
        test_jsonl=Path(args.test_jsonl) if args.test_jsonl else None,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text, flush=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
