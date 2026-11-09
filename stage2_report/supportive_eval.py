"""
Supportive consistency scoring for crash explanations.
事故解释一致性评分（规则脚本，非 VLM 自评）。

Per-video scores:
  - scene_ok / lighting_ok / weather_ok: binary match vs metadata detectors
  - context_supportive_mean: mean of the three context dims above
  - type_ok: Qwen collision_pattern vs GT type (optional; type is also a pipeline input)
  - supportive_mean: mean(scene, light, weather, type) — legacy composite; prefer context + veh_F1
  - vehicle_classes: set overlap recall/precision/F1 vs participant pseudo-GT

Aggregate (200 videos):
  - *_acc / parse_rate / causal_rate: fraction of videos where dim is True
  - context_supportive_mean: average of per-video context_supportive_mean
"""

from __future__ import annotations

import json
import re
from typing import Any

# --- keyword maps for fuzzy alignment ---

SCENE_KEYWORDS: dict[str, list[str]] = {
    "highway": ["highway", "freeway", "expressway", "motorway", "interstate"],
    "signalized_intersection": [
        "intersection", "crossroad", "crossroads", "traffic light",
        "signalized", "signal light", "junction",
    ],
    "simple_intersection": ["intersection", "crossroad", "junction", "unsignalized"],
    "grade_separated_intersection": [
        "interchange", "overpass", "underpass", "ramp", "merge", "grade-separated",
    ],
    "city_street": ["city street", "urban", "downtown", "street", "boulevard"],
    "parking_lot": ["parking", "lot", "garage"],
}

LIGHTING_KEYWORDS: dict[str, list[str]] = {
    "day": ["day", "daylight", "daytime", "sunny", "bright", "morning", "afternoon"],
    "night": ["night", "nighttime", "dark", "low light", "headlight", "streetlight"],
    "dusk": ["dusk", "twilight", "sunset", "dawn"],
}

WEATHER_KEYWORDS: dict[str, list[str]] = {
    "normal": ["clear", "dry", "normal", "fair", "no rain", "sunny"],
    "rain": ["rain", "rainy", "wet road", "wet pavement", "precipitation"],
    "snow": ["snow", "snowy", "icy", "ice"],
    "fog": ["fog", "foggy", "mist", "haze", "low visibility"],
}

VEHICLE_CLASS_KEYWORDS: dict[str, list[str]] = {
    "car": ["car", "sedan", "suv", "passenger vehicle", "passenger car", "hatchback", "van", "minivan"],
    "truck": ["truck", "semi", "semi-truck", "tractor-trailer", "lorry", "freight", "18-wheeler", "rig", "trailer truck"],
    "bus": ["bus", "coach", "transit bus", "school bus"],
    "motorcycle": ["motorcycle", "motorbike", "bike", "scooter", "moped"],
}

TYPE_KEYWORDS: dict[str, list[str]] = {
    "rear-end": ["rear-end", "rear end", "tail", "from behind", "following", "rear collision", "bumper"],
    "t-bone": ["t-bone", "t bone", "broadside", "side impact", "lateral", "perpendicular"],
    "head-on": ["head-on", "head on", "frontal", "opposite direction", "head-on collision"],
    "sideswipe": ["sideswipe", "side swipe", "lane change", "scraping", "glancing"],
    "single": [
        "single vehicle", "single-vehicle", "rollover", "off-road", "barrier",
        "pole", "tree", "guardrail", "solo",
    ],
}

SCENE_TO_LAYOUT: dict[str, str] = {
    "highway": "highway",
    "intersection": "signalized_intersection",  # generic bucket
    "city_street": "city_street",
    "parking": "parking_lot",
    "unknown": "",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = _norm(text)
    return any(k in t for k in keywords)


def _match_bucket(text: str, bucket_map: dict[str, list[str]]) -> str | None:
    t = _norm(text)
    for bucket, kws in bucket_map.items():
        if any(k in t for k in kws):
            return bucket
    return None


def _scene_compatible(pred_bucket: str | None, gt_layout: str) -> bool | None:
    if not pred_bucket or not gt_layout:
        return None
    gt = gt_layout.lower()
    pb = pred_bucket.lower()
    if pb == gt:
        return True
    # intersection family overlap
    inter = {"signalized_intersection", "simple_intersection", "grade_separated_intersection"}
    if pb in inter and gt in inter:
        return True
    if pb == "intersection" and gt in inter:
        return True
    if gt == "highway" and pb == "highway":
        return True
    return False


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    clean = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", clean, flags=re.IGNORECASE)
    if fence:
        clean = fence.group(1).strip()
    else:
        clean = re.sub(r"``[\s\S]*?``", "", clean, flags=re.IGNORECASE)
        end_think = re.search(r"``\s*", clean, flags=re.IGNORECASE)
        if end_think:
            clean = clean[end_think.end() :].strip()
    start = clean.find("{")
    if start < 0:
        return None

    def _try_load(chunk: str) -> dict[str, Any] | None:
        try:
            return json.loads(chunk)
        except json.JSONDecodeError:
            return None

    depth = 0
    for i in range(start, len(clean)):
        if clean[i] == "{":
            depth += 1
        elif clean[i] == "}":
            depth -= 1
            if depth == 0:
                parsed = _try_load(clean[start : i + 1])
                if parsed is not None:
                    return parsed
    # salvage truncated JSON (common when max_new_tokens cuts output)
    tail = clean[start:]
    for extra in ('"}]}', '"}]}', '"}', ']}', '}'):
        parsed = _try_load(tail + extra)
        if parsed is not None:
            return parsed
    return None


def flatten_explanation(parsed: dict[str, Any] | None, raw_text: str) -> str:
    if not parsed:
        return _norm(raw_text)
    parts: list[str] = []
    scene = parsed.get("scene")
    if isinstance(scene, dict):
        parts.append(str(scene.get("layout", "")))
        parts.append(str(scene.get("description", "")))
    elif scene:
        parts.append(str(scene))
    for key in ("lighting", "weather", "uncertainty"):
        if parsed.get(key):
            parts.append(str(parsed[key]))
    geom = parsed.get("geometry")
    if isinstance(geom, dict):
        parts.extend(str(v) for v in geom.values())
    for key in ("observations", "causal_hypothesis", "evidence"):
        val = parsed.get(key)
        if isinstance(val, list):
            parts.extend(str(x) for x in val)
        elif val:
            parts.append(str(val))
    parts.append(raw_text)
    return _norm(" ".join(parts))


def score_scene(text: str, gt_layout: str, parsed: dict | None = None) -> dict[str, Any]:
    pred_layout = None
    if parsed and isinstance(parsed.get("scene"), dict):
        pred_layout = parsed["scene"].get("layout")
    pred_bucket = _match_bucket(text, SCENE_KEYWORDS)
    if pred_layout:
        mapped = SCENE_TO_LAYOUT.get(str(pred_layout).lower(), str(pred_layout).lower())
        if mapped:
            pred_bucket = mapped if mapped != "signalized_intersection" else pred_bucket or "intersection"
    ok = _scene_compatible(pred_bucket, gt_layout)
    if ok is None and pred_layout:
        ok = _scene_compatible(str(pred_layout).lower(), gt_layout)
    return {"ok": ok, "pred_bucket": pred_bucket, "gt": gt_layout, "pred_layout_field": pred_layout}


def score_lighting(text: str, gt_day_time: str, parsed: dict | None = None) -> dict[str, Any]:
    pred = None
    if parsed and parsed.get("lighting"):
        pred = str(parsed["lighting"]).lower()
    else:
        pred = _match_bucket(text, LIGHTING_KEYWORDS)
    gt = (gt_day_time or "").lower()
    ok = None
    if pred and gt:
        ok = pred == gt or (pred in gt) or (gt in pred)
    return {"ok": ok, "pred": pred, "gt": gt}


def score_weather(text: str, gt_weather: str, parsed: dict | None = None) -> dict[str, Any]:
    pred = None
    if parsed and parsed.get("weather"):
        pred = str(parsed["weather"]).lower()
    else:
        pred = _match_bucket(text, WEATHER_KEYWORDS)
    gt = (gt_weather or "normal").lower()
    ok = None
    if pred and gt:
        if gt == "normal":
            ok = pred in ("normal", "clear", "unknown") or _contains_any(text, WEATHER_KEYWORDS["normal"])
        else:
            ok = pred == gt or _contains_any(text, WEATHER_KEYWORDS.get(gt, [gt]))
    return {"ok": ok, "pred": pred, "gt": gt}


def score_type(text: str, gt_type: str, parsed: dict | None = None) -> dict[str, Any]:
    pred = None
    if parsed and isinstance(parsed.get("geometry"), dict):
        pred = parsed["geometry"].get("collision_pattern")
    if not pred:
        pred = _match_bucket(text, TYPE_KEYWORDS)
    gt = (gt_type or "").lower()
    ok = None
    if pred and gt:
        pred_n = str(pred).lower().replace("_", "-")
        ok = pred_n == gt or _contains_any(text, TYPE_KEYWORDS.get(gt, [gt]))
    return {"ok": ok, "pred": pred, "gt": gt}


def score_explanation(
    raw_text: str,
    *,
    gt_layout: str,
    gt_day_time: str,
    gt_weather: str,
    gt_type: str,
    structured: bool,
) -> dict[str, Any]:
    parsed = extract_json_from_text(raw_text) if structured else None
    flat = flatten_explanation(parsed, raw_text)

    scene = score_scene(flat, gt_layout, parsed)
    lighting = score_lighting(flat, gt_day_time, parsed)
    weather = score_weather(flat, gt_weather, parsed)
    crash_type = score_type(flat, gt_type, parsed)

    context_dims = [scene, lighting, weather]
    all_dims = context_dims + [crash_type]
    context_scored = [d["ok"] for d in context_dims if d["ok"] is not None]
    all_scored = [d["ok"] for d in all_dims if d["ok"] is not None]
    # Per-video context score: fraction of {scene, light, weather} that match GT metadata.
    # 单条视频：scene/light/weather 三维中判为一致的比例。
    context_supportive_mean = (
        sum(1 for x in context_scored if x) / len(context_scored) if context_scored else None
    )
    # Per-video supportive_mean: context dims + type_ok (4-way). Legacy metric.
    # 单条视频：scene+light+weather+type 四维一致比例（含 type，论文主表建议用 context）。
    supportive_mean = sum(1 for x in all_scored if x) / len(all_scored) if all_scored else None

    parse_ok = parsed is not None if structured else True
    causal = has_causal_claim(parsed) if structured else None
    vehicle_classes = extract_vehicle_classes(parsed, flat)
    return {
        "parse_ok": parse_ok,
        "parsed": parsed,
        "scene_ok": scene["ok"],
        "lighting_ok": lighting["ok"],
        "weather_ok": weather["ok"],
        "type_ok": crash_type["ok"],
        "has_causal_hypothesis": causal,
        "context_supportive_mean": context_supportive_mean,
        "supportive_mean": supportive_mean,
        "n_context_dims": len(context_scored),
        "n_scored_dims": len(all_scored),
        "vehicle_classes_pred": vehicle_classes,
        "details": {"scene": scene, "lighting": lighting, "weather": weather, "type": crash_type},
    }


def extract_vehicle_classes(parsed: dict[str, Any] | None, flat_text: str) -> list[str]:
    found: set[str] = set()
    if parsed:
        vehicles = parsed.get("vehicles")
        if isinstance(vehicles, list):
            for v in vehicles:
                if isinstance(v, dict):
                    cls = str(v.get("class", "")).lower()
                    if cls and cls != "unknown":
                        found.add(cls)
        for key in ("observations", "geometry"):
            val = parsed.get(key)
            if isinstance(val, dict):
                val = val.get("spatial_description", "")
            if isinstance(val, list):
                val = " ".join(str(x) for x in val)
            if val:
                for cls, kws in VEHICLE_CLASS_KEYWORDS.items():
                    if any(k in str(val).lower() for k in kws):
                        found.add(cls)
    for cls, kws in VEHICLE_CLASS_KEYWORDS.items():
        if any(k in flat_text for k in kws):
            found.add(cls)
    return sorted(found)


def score_vehicle_classes(pred_classes: list[str], gt_classes: list[str]) -> dict[str, Any]:
    """Overlap-based score when weak GT is available (e.g. from detector pseudo-labels).
    
    Returns:
        recall: fraction of GT classes mentioned by Qwen
        precision: fraction of Qwen classes that are in GT
        f1: harmonic mean
    """
    if not gt_classes:
        return {"ok": None, "pred": pred_classes, "gt": gt_classes, "recall": None, "precision": None, "f1": None}
    pred = set(pred_classes)
    gt = set(gt_classes)
    if not pred:
        return {"ok": False, "pred": pred_classes, "gt": gt_classes, "recall": 0.0, "precision": 0.0, "f1": 0.0}
    
    overlap = pred & gt
    recall = len(overlap) / len(gt) if gt else 0.0
    precision = len(overlap) / len(pred) if pred else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
    ok = recall >= 0.5
    return {
        "ok": ok,
        "pred": pred_classes,
        "gt": gt_classes,
        "overlap": sorted(overlap),
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def has_causal_claim(parsed: dict[str, Any] | None) -> bool | None:
    if not parsed:
        return None
    hyp = parsed.get("causal_hypothesis")
    if isinstance(hyp, list):
        return any(str(x).strip() for x in hyp)
    return bool(str(hyp or "").strip())


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-video scores to dataset-level rates.

    scene_acc etc. = (# videos with *_ok True) / (# videos with a judgment).
    context_supportive_mean = mean of per-video context_supportive_mean values.
    supportive_mean = mean of per-video supportive_mean (includes type; see README).
    """

    def rate(key: str) -> float | None:
        vals = [r[key] for r in rows if r.get(key) is True or r.get(key) is False]
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals)

    supportive = [r["supportive_mean"] for r in rows if r.get("supportive_mean") is not None]
    context = []
    for r in rows:
        if r.get("context_supportive_mean") is not None:
            context.append(r["context_supportive_mean"])
            continue
        dims = [r.get("scene_ok"), r.get("lighting_ok"), r.get("weather_ok")]
        scored = [d for d in dims if d is not None]
        if scored:
            context.append(sum(1 for x in scored if x) / len(scored))
    causal = [
        r["has_causal_hypothesis"]
        for r in rows
        if r.get("has_causal_hypothesis") is True or r.get("has_causal_hypothesis") is False
    ]
    return {
        "n": len(rows),
        "parse_rate": sum(1 for r in rows if r.get("parse_ok")) / len(rows) if rows else 0,
        "scene_acc": rate("scene_ok"),
        "lighting_acc": rate("lighting_ok"),
        "weather_acc": rate("weather_ok"),
        "type_acc": rate("type_ok"),
        "context_supportive_mean": sum(context) / len(context) if context else None,
        "supportive_mean": sum(supportive) / len(supportive) if supportive else None,
        "causal_rate": sum(1 for x in causal if x) / len(causal) if causal else None,
    }
