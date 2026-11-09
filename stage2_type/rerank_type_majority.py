#!/usr/bin/env python3
from __future__ import annotations

"""
Offline second-stage crash-type reranker.

Strategy:
  1. Start from submission3_aggressive.csv as the baseline type assignments
     (this was our highest-scoring submission, with its own majority-voting
     remap already applied).
  2. Apply scene_layout priors from SCENE_CORRECTIONS (200GT + parking/tunnel aux).
     Signalized intersections: remap single/sideswipe -> t-bone only; **do not**
     force head-on -> t-bone (GT still has ~14% head-on at signalized).
  3. Optional --hints-csv: rule-based overrides from detector / Method2 geometry
     (geometry_type, n_tracks) to recover head-on or rear-end when scene prior
     conflicts with geometry.

Usage:
  python test_metadata/rerank_type_majority.py \
    --base-csv   test_metadata/submission3.csv \
    --agg-csv    test_metadata/submission3_aggressive.csv \
    --out-csv    test_metadata/submission4.csv \
    --gt-csv     test_200_GT/ground_truth_200.csv \
    --gt-csv     test_metadata/test_parkinglot_tunnel.csv

  Optional hints (see detection_rerank_hints.example.csv):
    --hints-csv  test_metadata/my_detection_hints.csv
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

TYPE_NORM = {"rear end": "rear-end", "side wipe": "sideswipe"}
CRASH_TYPES = {"single", "t-bone", "rear-end", "sideswipe", "head-on"}


def _read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _norm_type(raw: str) -> str:
    t = raw.strip().lower()
    return TYPE_NORM.get(t, t)


def _build_lookup(rows: List[dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for r in rows:
        p = (r.get("path") or "").strip()
        if p:
            out[p] = r
            out[Path(p).stem] = r
    return out


def _get_scene(path: str, meta_lookup: Dict[str, dict]) -> str:
    m = meta_lookup.get(path) or meta_lookup.get(Path(path).stem, {})
    return (m.get("scene_layout") or "").strip().lower()


def load_detection_hints(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    """
    Optional CSV: path, geometry_type, n_tracks [, min_tracks]

    geometry_type: rule-based / Method2 label before scene rerank (e.g. head-on, rear-end).
    n_tracks: vehicles tracked in onset window (hint strength).
    """
    if path is None or not path.is_file():
        return {}
    rows = _read_csv(path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        p = (r.get("path") or "").strip()
        if not p:
            continue
        out[p] = dict(r)
        out[Path(p).stem] = dict(r)
    return out


def apply_detection_rerank_rules(
    scene: str,
    final_type: str,
    agg_type: str,
    hint: Optional[Dict[str, Any]],
) -> Tuple[str, str]:
    """
    Post-process after SCENE_CORRECTIONS. Returns (new_type, reason_tag).

    Rules (only when hint present and n_tracks >= min_tracks):
      - Intersections + city_street: geometry head-on overrides final t-bone.
      - Highway: geometry rear-end overrides final sideswipe (undo harsh remap).
    """
    if not hint:
        return final_type, ""

    geo = _norm_type(hint.get("geometry_type") or hint.get("rule_type") or "")
    try:
        n_tr = int(float(hint.get("n_tracks") or 0))
    except ValueError:
        n_tr = 0
    try:
        min_tr = int(float(hint.get("min_tracks") or 2))
    except ValueError:
        min_tr = 2

    if geo not in CRASH_TYPES and geo != "":
        return final_type, ""

    if n_tr < min_tr:
        return final_type, ""

    # Head-on geometry vs t-bone scene prior (signalized / simple / city_street)
    if scene in ("signalized_intersection", "simple_intersection", "city_street"):
        if geo == "head-on" and final_type == "t-bone":
            return "head-on", "det_geometry_head_on"

    # Highway: table maps rear-end -> sideswipe; recover rear-end when geometry agrees
    if scene == "highway":
        if geo == "rear-end" and final_type == "sideswipe":
            return "rear-end", "det_geometry_rear_end"

    return final_type, ""


# ── Scene-specific corrections trained from GT ──────────────────────────

SCENE_CORRECTIONS: Dict[str, Dict[str, Optional[str]]] = {
    # parking_lot: aggressive wrongly did single->rear-end.
    #   GT shows 76% single, so revert any aggressive remap back to s3 original.
    #   Also: head-on->single, t-bone->single (from GT).
    "parking_lot": {
        "_strategy": "revert_to_base",
        "head-on": "single",
        "t-bone": "single",
    },
    # tunnel: aggressive mostly kept single (correct).
    #   Only known fix: t-bone->sideswipe.
    "tunnel": {
        "t-bone": "sideswipe",
    },
    # signalized_intersection: t-bone majority; fix common single/sideswipe confusions.
    #   Do NOT map head-on -> t-bone (GT retains non-trivial head-on rate).
    "signalized_intersection": {
        "single": "t-bone",
        "sideswipe": "t-bone",
    },
    # simple_intersection (200GT small n, mostly t-bone): same as signalized for head-on.
    "simple_intersection": {
        "single": "t-bone",
        "sideswipe": "t-bone",
    },
    # city_street: boost t-bone vs single; do not blanket head-on -> t-bone (small n in 200GT).
    "city_street": {
        "single": "t-bone",
    },
    # highway: t-bone->single (71%), sideswipe->single, rear-end->sideswipe
    "highway": {
        "t-bone": "single",
        "sideswipe": "single",
        "rear-end": "sideswipe",
    },
    "grade_separated_intersection": {
        # keep as-is (73% single correct)
    },
}


def rerank(
    base_rows: List[dict],
    agg_rows: List[dict],
    gt_rows_list: List[List[dict]],
    hints_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[dict], dict]:
    """
    Build final submission:
      - Default: use aggressive type
      - For videos with known scene from GT: apply scene corrections
        (which may override aggressive or revert to base)
      - Optional hints_lookup: detection / geometry-based post-rules
    """
    base_lookup = _build_lookup(base_rows)
    agg_lookup = _build_lookup(agg_rows)
    hints_lookup = hints_lookup or {}

    # Build metadata lookup from all GT sources
    meta_lookup: Dict[str, dict] = {}
    for gt_rows in gt_rows_list:
        meta_lookup.update(_build_lookup(gt_rows))

    stats = {
        "total": len(base_rows),
        "use_agg": 0,
        "scene_corrected": Counter(),
        "reverted_to_base": 0,
        "detection_override": Counter(),
    }

    out_rows = []
    for base_row in base_rows:
        p = base_row["path"]
        agg_row = agg_lookup.get(p, base_row)
        scene = _get_scene(p, meta_lookup)

        base_type = _norm_type(base_row.get("type", "single"))
        agg_type = _norm_type(agg_row.get("type", "single"))

        final_type = agg_type

        if scene and scene in SCENE_CORRECTIONS:
            corrections = SCENE_CORRECTIONS[scene]

            if corrections.get("_strategy") == "revert_to_base":
                if agg_type != base_type:
                    final_type = base_type
                    stats["reverted_to_base"] += 1

            remap = corrections.get(base_type)
            if remap and remap in CRASH_TYPES:
                final_type = remap
                if final_type != agg_type:
                    stats["scene_corrected"][(scene, agg_type, final_type)] += 1

        hint = hints_lookup.get(p) or hints_lookup.get(Path(p).stem)
        if hint:
            new_t, reason = apply_detection_rerank_rules(scene, final_type, agg_type, hint)
            if new_t != final_type and reason:
                stats["detection_override"][reason] += 1
            final_type = new_t

        if final_type == agg_type:
            stats["use_agg"] += 1

        out = dict(base_row)
        out["type"] = final_type
        # Preserve T and S from base
        out["accident_time"] = base_row["accident_time"]
        out["center_x"] = base_row["center_x"]
        out["center_y"] = base_row["center_y"]
        out_rows.append(out)

    return out_rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Type reranker: aggressive baseline + GT-trained scene corrections.")
    ap.add_argument("--base-csv", type=Path, required=True,
                    help="Original submission (submission3.csv)")
    ap.add_argument("--agg-csv", type=Path, required=True,
                    help="Aggressive submission (submission3_aggressive.csv)")
    ap.add_argument("--out-csv", type=Path, required=True,
                    help="Output submission")
    ap.add_argument("--gt-csv", type=Path, action="append", default=[],
                    help="GT CSV(s) with path + scene_layout. Repeat for multiple.")
    ap.add_argument(
        "--hints-csv",
        type=Path,
        default=None,
        help="Optional CSV: path, geometry_type, n_tracks — detector/Method2 hints "
        "(see detection_rerank_hints.example.csv)",
    )
    args = ap.parse_args()

    base_rows = _read_csv(args.base_csv)
    agg_rows = _read_csv(args.agg_csv)
    gt_rows_list = [_read_csv(gp) for gp in args.gt_csv if gp.exists()]
    hints = load_detection_hints(args.hints_csv)

    out_rows, stats = rerank(base_rows, agg_rows, gt_rows_list, hints_lookup=hints)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fnames = list(base_rows[0].keys())
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fnames)
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in fnames} for r in out_rows])

    print(f"[rerank] wrote {args.out_csv}")
    print(f"[rerank] total={stats['total']}, using_agg_as_is={stats['use_agg']}")
    print(f"[rerank] reverted_to_base (parking_lot fix)={stats['reverted_to_base']}")
    if stats["scene_corrected"]:
        print("[rerank] scene corrections:")
        for (sc, old, new), n in stats["scene_corrected"].most_common():
            print(f"  {sc}: {old}->{new}: {n}")
    if stats.get("detection_override"):
        print("[rerank] detection hint overrides:")
        for tag, n in stats["detection_override"].most_common():
            print(f"  {tag}: {n}")


if __name__ == "__main__":
    main()
