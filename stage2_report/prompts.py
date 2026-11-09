"""
Prompt variants for crash explanation experiments.
事故解释 prompt 变体（CCTV 固定机位 + JSON 结构化输出）。

v6: pipeline when/where/what + 通用 CCTV 指令
v7: v6 + 只描述事故参与车辆（PARTICIPANT_FOCUS）
"""

from __future__ import annotations

from typing import Any

CCTV_JSON_SCHEMA_HINT = """Reply with ONLY valid JSON (no markdown fences):
{
  "scene": {"layout": "highway|intersection|city_street|parking|unknown", "description": "road layout, lanes, fixed-camera viewpoint"},
  "lighting": "day|night|dusk|unknown",
  "weather": "clear|rain|snow|fog|unknown",
  "geometry": {
    "collision_pattern": "rear-end|t-bone|head-on|sideswipe|single|unknown",
    "n_vehicles": 0,
    "spatial_description": "approach angles, contact point, motion before/after impact"
  },
  "vehicles": [
    {"class": "car|truck|bus|motorcycle|unknown", "role": "e.g. striking vehicle / struck vehicle / single vehicle"}
  ],
  "observations": ["objective visible facts about scene, traffic, and collision dynamics"],
  "causal_hypothesis": ["likely contributing factors / why the accident happened"],
  "evidence": ["frame/time-specific visual cues supporting each hypothesis"],
  "confidence": 0.0,
  "uncertainty": "what cannot be determined from this CCTV view"
}"""

CCTV_INSTRUCTIONS = (
    "Camera: fixed overhead/side CCTV traffic camera (NOT dashcam, NOT ego-vehicle view).\n"
    "The collision is known to have occurred; your job is to explain WHY it likely happened.\n"
    "Strictly separate:\n"
    "  OBSERVATIONS = only what is directly visible (scene, lighting, weather, vehicle motion, contact).\n"
    "  CAUSAL HYPOTHESIS = plausible accident causes inferred from those observations.\n"
    "Do NOT describe vehicle brand, make, model, or color unless absolutely necessary for the mechanism.\n"
    "Do NOT invent traffic-signal state, speed, or driver intent unless clearly visible.\n"
    "If the cause cannot be determined, leave causal_hypothesis empty and explain in uncertainty.\n"
    "Prioritize accident mechanism: following too close, failure to yield, unsafe lane change, "
    "loss of control, visibility, road geometry, etc."
)


def build_frame_labels(timestamps: list[float]) -> str:
    return ", ".join(f"Frame {i + 1} = {t:.1f}s" for i, t in enumerate(timestamps))


def prompt_v1_freeform(timestamps: list[float], duration: float, **_: Any) -> str:
    labels = build_frame_labels(timestamps)
    return (
        "You are analyzing dashcam/traffic footage of a traffic accident.\n"
        f"Video duration: {duration:.1f}s. Frames shown at: {labels}.\n\n"
        "Explain what happened in this accident. Cover:\n"
        "1) Scene type (highway, intersection, etc.)\n"
        "2) Lighting conditions (day/night)\n"
        "3) Weather if visible\n"
        "4) Collision geometry (how vehicles contacted)\n"
        "5) Likely contributing factors (why it may have happened)\n\n"
        "Write 3-6 sentences. Base claims only on visible evidence."
    )


def prompt_v2_structured(timestamps: list[float], duration: float, **_: Any) -> str:
    labels = build_frame_labels(timestamps)
    return (
        "You are a traffic-accident analyst reviewing CCTV footage of a confirmed collision.\n"
        f"Video duration: {duration:.1f}s. {len(timestamps)} frames around the collision at: {labels}.\n\n"
        + CCTV_INSTRUCTIONS
        + "\n\n"
        + CCTV_JSON_SCHEMA_HINT
    )


def prompt_v3_multiframe(timestamps: list[float], duration: float, **_: Any) -> str:
    labels = build_frame_labels(timestamps)
    return (
        "You are an accident analyst reviewing multiple frames around the collision.\n"
        f"Video duration: {duration:.1f}s. {len(timestamps)} frames at: {labels}.\n\n"
        "Use temporal changes across frames to infer collision dynamics.\n"
        "Focus on scene, lighting, weather, geometry, then cautious causal hypotheses.\n\n"
        + CCTV_JSON_SCHEMA_HINT
    )


def _pipeline_prior_block(
    *,
    pred_time: float = 0.0,
    pred_type: str = "",
    pred_cx: float | None = None,
    pred_cy: float | None = None,
) -> str:
    where = "unknown"
    if pred_cx is not None and pred_cy is not None:
        where = f"normalized ({pred_cx:.3f}, {pred_cy:.3f}), image top-left=(0,0) bottom-right=(1,1)"
    return (
        "Pipeline predictions (may be wrong — verify and correct using CCTV frames):\n"
        f"  WHEN  — estimated collision time: {pred_time:.2f}s\n"
        f"  WHAT  — estimated crash type: {pred_type or 'unknown'}\n"
        f"  WHERE — estimated collision point: {where}\n"
    )


def prompt_v4_prior_assisted(
    timestamps: list[float],
    duration: float,
    *,
    pred_type: str = "",
    pred_time: float = 0.0,
    pred_cx: float | None = None,
    pred_cy: float | None = None,
    **_: Any,
) -> str:
    labels = build_frame_labels(timestamps)
    return (
        "You are a traffic-accident analyst reviewing CCTV footage of a confirmed collision.\n"
        f"Video duration: {duration:.1f}s. {len(timestamps)} frames around the predicted collision at: {labels}.\n\n"
        + _pipeline_prior_block(
            pred_time=pred_time, pred_type=pred_type, pred_cx=pred_cx, pred_cy=pred_cy
        )
        + "\n"
        + CCTV_INSTRUCTIONS
        + "\nUse WHEN/WHAT/WHERE hints to focus on the collision, but correct any wrong hints.\n\n"
        + CCTV_JSON_SCHEMA_HINT
    )


def prompt_v6_cctv_pipeline(
    timestamps: list[float],
    duration: float,
    *,
    pred_type: str = "",
    pred_time: float = 0.0,
    pred_cx: float | None = None,
    pred_cy: float | None = None,
    **_: Any,
) -> str:
    """v2 CCTV prompt + full sub5 when/where/what priors; frames already at pred onset."""
    labels = build_frame_labels(timestamps)
    return (
        "You are a traffic-accident analyst reviewing CCTV footage of a confirmed collision.\n"
        f"Video duration: {duration:.1f}s. {len(timestamps)} frames centered on the pipeline-predicted "
        f"collision time. Frame timestamps: {labels}.\n\n"
        + _pipeline_prior_block(
            pred_time=pred_time, pred_type=pred_type, pred_cx=pred_cx, pred_cy=pred_cy
        )
        + "\n"
        + CCTV_INSTRUCTIONS
        + "\nYour explanation should be consistent with visible evidence. "
        "If pipeline WHAT/WHEN/WHERE conflicts with the video, trust the video and note the mismatch in uncertainty.\n\n"
        + CCTV_JSON_SCHEMA_HINT
    )


PARTICIPANT_FOCUS = (
    "CRASH PARTICIPANTS ONLY:\n"
    "  Focus on the vehicle(s) directly involved in the collision at the pipeline-predicted WHERE point.\n"
    "  For single-vehicle crashes: describe only that one vehicle.\n"
    "  For multi-vehicle crashes: describe only the two vehicles that collide (striking + struck).\n"
    "  Do NOT mention background traffic, parked vehicles, or vehicles passing by that are not involved.\n"
    "  In vehicles[], list ONLY crash participants with their class (car/truck/bus/motorcycle) and role.\n"
    "  In observations[], describe participant motion and contact — not unrelated vehicles in the scene.\n"
)


def prompt_v7_cctv_participants(
    timestamps: list[float],
    duration: float,
    *,
    pred_type: str = "",
    pred_time: float = 0.0,
    pred_cx: float | None = None,
    pred_cy: float | None = None,
    **_: Any,
) -> str:
    """v6 + explicit focus on crash participants only (not all vehicles in frame)."""
    labels = build_frame_labels(timestamps)
    return (
        "You are a traffic-accident analyst reviewing CCTV footage of a confirmed collision.\n"
        f"Video duration: {duration:.1f}s. {len(timestamps)} frames centered on the pipeline-predicted "
        f"collision time. Frame timestamps: {labels}.\n\n"
        + _pipeline_prior_block(
            pred_time=pred_time, pred_type=pred_type, pred_cx=pred_cx, pred_cy=pred_cy
        )
        + "\n"
        + PARTICIPANT_FOCUS
        + "\n"
        + CCTV_INSTRUCTIONS
        + "\nYour explanation should be consistent with visible evidence. "
        "If pipeline WHAT/WHEN/WHERE conflicts with the video, trust the video and note the mismatch in uncertainty.\n\n"
        + CCTV_JSON_SCHEMA_HINT
    )


def prompt_v5_geometry_focus(timestamps: list[float], duration: float, **_: Any) -> str:
    labels = build_frame_labels(timestamps)
    return (
        "You are analyzing collision geometry in traffic accident footage.\n"
        f"Video duration: {duration:.1f}s. Frames at: {labels}.\n\n"
        "Prioritize: vehicle count, approach angles, contact point, motion before/after impact.\n"
        "Then briefly note scene, lighting, weather.\n"
        "Causal hypotheses must follow from geometry.\n\n"
        + CCTV_JSON_SCHEMA_HINT
    )


PROMPT_REGISTRY: dict[str, Any] = {
    "v1_freeform": {
        "fn": prompt_v1_freeform,
        "structured": False,
        "desc": "Open-ended narrative, no JSON",
    },
    "v2_structured": {
        "fn": prompt_v2_structured,
        "structured": True,
        "desc": "CCTV multi-frame JSON, observation vs causal hypothesis",
    },
    "v3_multiframe": {
        "fn": prompt_v3_multiframe,
        "structured": True,
        "desc": "Multi-frame temporal structured JSON",
    },
    "v4_prior_assisted": {
        "fn": prompt_v4_prior_assisted,
        "structured": True,
        "desc": "CCTV JSON + sub5 when/what/where hints (5-frame, pred onset)",
    },
    "v6_cctv_pipeline": {
        "fn": prompt_v6_cctv_pipeline,
        "structured": True,
        "desc": "CCTV 9-frame + sub5 when/what/where (all pred, no GT leak)",
    },
    "v7_cctv_participants": {
        "fn": prompt_v7_cctv_participants,
        "structured": True,
        "desc": "v6 + crash-participant focus (exclude background traffic)",
    },
    "v5_geometry_focus": {
        "fn": prompt_v5_geometry_focus,
        "structured": True,
        "desc": "Geometry-first structured JSON",
    },
}
