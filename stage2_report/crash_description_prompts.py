"""
0618_Crash-description prompts with human-correction style guidelines.
事故描述 prompt：强调冲突机理（如交叉口 t-bone = turning vs through）。

Two prompt families / 两类 prompt:
  - build_gt_prompt_v2: gold-200 pipeline (Qwen), generic hand-written few-shot.
    gold-200 流程（Qwen），使用手写通用 few-shot。
  - build_silver_gt_prompt: silver pipeline (Gemini), uses REAL gold-200 records
    as few-shot examples and asks ONLY for observation/explanation/summary/participants.
    silver 流程（Gemini），用真实 gold-200 记录作 few-shot，仅生成
    observation/explanation/summary/participants（固定字段直接给定，不重新识别）。
"""

from __future__ import annotations

import json
import random
from typing import Any

from prompts import build_frame_labels

# Few-shot examples distilled from manual review (turning vs through at intersections).
# 人工核查样例：交叉口 t-bone 必须写清 turning path vs through path。
FEW_SHOT_EXAMPLES = """
EXAMPLE A — t-bone at signalized intersection (turning vs straight):
{
  "observation": [
    "A dark sedan enters the intersection turning left from the bottom of the frame.",
    "A second car travels straight from the upper-left toward the intersection.",
    "The straight-traveling car strikes the driver-side front of the turning sedan."
  ],
  "explanation": "The turning sedan crossed the through path of the straight-traveling car without yielding, causing a t-bone collision.",
  "summary": "At a signalized intersection, a left-turning sedan was t-boned by a vehicle traveling straight through the junction."
}

EXAMPLE B — t-bone (left turn vs straight):
{
  "observation": [
    "The white sedan travels straight through the intersection.",
    "The black sedan executes a left turn across the white sedan's path.",
    "The front of the black sedan collides with the driver-side of the white sedan."
  ],
  "explanation": "The black sedan failed to yield while turning left across the through path of the white sedan.",
  "summary": "A black sedan turning left t-boned a white sedan going straight at an intersection."
}

EXAMPLE C — rear-end on highway:
{
  "observation": [
    "A red semi-truck follows traffic in the right lane.",
    "A sedan moves into the emergency lane ahead of the truck.",
    "The truck merges into the emergency lane and collides with the rear of the sedan."
  ],
  "explanation": "The truck did not stop in time and rear-ended the sedan after both entered the emergency lane.",
  "summary": "On a highway, a truck rear-ended a sedan in the emergency lane."
}

EXAMPLE D — single-vehicle:
{
  "observation": [
    "A car travels straight in the right highway lane.",
    "The car loses control and drifts off the roadway.",
    "The car collides with the roadside barrier."
  ],
  "explanation": "The car lost control and struck the roadside barrier.",
  "summary": "A single car lost control on the highway and hit the barrier."
}
"""

COLLISION_MECHANISM_GUIDE = """
CRITICAL RULES (from human review — follow strictly):

1. The collision IS confirmed. Describe what happened; NEVER say "no collision visible" or "false positive".

2. Focus on CONFLICT GEOMETRY — who moves how, and which paths cross:
   - t-bone at intersection: almost always TURNING path vs THROUGH/STRAIGHT path.
     State clearly: "Vehicle A turns left/right" AND "Vehicle B travels straight".
     Explanation: turning/merging vehicle failed to yield to through traffic.
   - rear-end: following too close, failed to stop, lane merge into lead vehicle, or hit stopped car on shoulder.
   - sideswipe: unsafe lane change / drift into adjacent lane.
   - head-on at intersection: opposing paths or turn vs straight frontal conflict.
   - single-vehicle: loss of control → barrier, pole, median, trees, or off-road.

3. OBSERVATION (1–3 bullets):
   - Bullet 1–2: each participant's approach path (straight / turn left / merge from ramp / follow in lane).
   - Final bullet: contact point and outcome.
   - Use roles when helpful: "the turning car", "the through car", "the following car".

4. EXPLANATION: one concise sentence naming the conflict (e.g., turning vs straight, merge vs through).

5. SUMMARY: one sentence restating scene + conflict + outcome (optional but preferred).

6. Do NOT invent signal color or exact speed. Do NOT describe background traffic unrelated to the crash.
"""


def build_gt_prompt_v2(
    timestamps: list[float],
    duration: float,
    fixed: dict[str, Any],
    accident_time: float,
) -> str:
    """Prompt for crash understanding with mechanism-focused style."""
    # 强调冲突机理的 V2 prompt
    labels = build_frame_labels(timestamps)
    participants_txt = ", ".join(fixed["participants"]) if fixed["participants"] else "unknown (verify from video)"
    n_participants = len(fixed["participants"]) if fixed["participants"] else (
        1 if fixed["_collision_type_gt"] == "single" else 2
    )
    collision_type = fixed["collision_type"]
    scene = fixed["scene"]

    tbone_hint = ""
    if collision_type == "t-bone" and "intersection" in scene:
        tbone_hint = (
            "\nT-BONE INTERSECTION HINT: Identify which vehicle is TURNING and which is traveling "
            "STRIGHT (through path). The core conflict is turning path crossing through path.\n"
        )

    return (
        "You are annotating CCTV traffic-camera footage for a crash-understanding dataset.\n"
        f"Video duration: {duration:.1f}s. Collision time: {accident_time:.2f}s.\n"
        f"{len(timestamps)} frames around collision: {labels}.\n\n"
        "LOCKED METADATA (do NOT change):\n"
        f"  scene: {scene}\n"
        f"  weather: {fixed['weather']}\n"
        f"  lighting: {fixed['lighting']}\n"
        f"  collision_type: {collision_type}\n"
        f"  participants ({n_participants}): {participants_txt}\n"
        + tbone_hint
        + "\n"
        + COLLISION_MECHANISM_GUIDE
        + "\n"
        + FEW_SHOT_EXAMPLES
        + "\n\nNow annotate THIS video. Reply with ONLY valid JSON:\n"
        "{\n"
        '  "observation": ["path/motion of each participant", "...", "contact and result"],\n'
        '  "explanation": "one sentence: core conflict mechanism",\n'
        '  "summary": "one sentence overview"\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Silver pipeline: real gold-200 records as few-shot examples (Gemini).
# silver 流程：用真实 gold-200 记录作为 few-shot 示例（Gemini）。
# ---------------------------------------------------------------------------

# Vehicle classes Gemini is allowed to report (kept consistent with gold labels).
# 允许 Gemini 报告的车辆类别（与 gold 标注保持一致）。
PARTICIPANT_VOCAB = ["car", "truck", "bus", "motorcycle", "bicycle", "pedestrian"]


def select_few_shot(
    gold_records: list[dict[str, Any]],
    collision_type: str,
    *,
    k_same: int = 2,
    k_other: int = 1,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Pick few-shot examples from human-reviewed gold-200 records.
    从人工核查的 gold-200 记录中挑选 few-shot 示例。

    Strategy / 策略:
      - k_same examples of the SAME collision_type (teaches the right mechanism wording).
        k_same 条同类型示例（教会正确的机理表述）。
      - k_other examples of OTHER types (keeps output format general, avoids over-fit).
        k_other 条其他类型示例（保持输出格式通用，避免过拟合）。
    Only well-formed records (non-empty obs/explanation/summary) are eligible.
    仅选用字段完整（obs/explanation/summary 非空）的记录。
    """
    rng = random.Random(seed)

    def _well_formed(r: dict[str, Any]) -> bool:
        return bool(r.get("observation")) and bool(r.get("explanation")) and bool(r.get("summary"))

    pool = [r for r in gold_records if _well_formed(r)]
    same = [r for r in pool if r.get("collision_type") == collision_type]
    other = [r for r in pool if r.get("collision_type") != collision_type]
    rng.shuffle(same)
    rng.shuffle(other)
    return same[:k_same] + other[:k_other]


def _render_example(rec: dict[str, Any]) -> str:
    """Render one gold record as a compact few-shot block (known fields + annotation)."""
    # 将一条 gold 记录渲染为紧凑 few-shot 块（已知字段 + 标注答案）
    known = (
        f"  scene={rec.get('scene')}, weather={rec.get('weather')}, "
        f"lighting={rec.get('lighting')}, collision_type={rec.get('collision_type')}"
    )
    answer = json.dumps(
        {
            "participants": rec.get("participants") or [],
            "observation": rec.get("observation") or [],
            "explanation": rec.get("explanation") or "",
            "summary": rec.get("summary") or "",
        },
        ensure_ascii=False,
        indent=2,
    )
    return f"KNOWN:\n{known}\nANNOTATION:\n{answer}"


def build_silver_gt_prompt(
    timestamps: list[float],
    duration: float,
    fixed: dict[str, Any],
    accident_time: float,
    few_shot_records: list[dict[str, Any]],
) -> str:
    """
    Silver-GT prompt: fixed metadata is GIVEN; Gemini only writes
    observation / explanation / summary / participants, guided by real gold few-shot.
    silver GT prompt：固定 metadata 已给定；Gemini 仅生成
    observation / explanation / summary / participants，由真实 gold few-shot 引导。
    """
    labels = build_frame_labels(timestamps)
    collision_type = fixed["collision_type"]
    scene = fixed["scene"]

    # Few-shot block from real human-reviewed annotations.
    # 来自真实人工核查标注的 few-shot 块。
    examples_txt = "\n\n".join(
        f"--- EXAMPLE {i + 1} ---\n{_render_example(rec)}"
        for i, rec in enumerate(few_shot_records)
    )

    tbone_hint = ""
    if collision_type == "t-bone" and "intersection" in scene:
        tbone_hint = (
            "\nT-BONE INTERSECTION HINT: identify which vehicle is TURNING and which travels "
            "STRAIGHT (through path); the core conflict is turning path crossing through path.\n"
        )

    vocab_txt = ", ".join(PARTICIPANT_VOCAB)

    return (
        "You are annotating CCTV traffic-camera footage for a crash-understanding dataset.\n"
        "The fixed metadata below is ALREADY KNOWN and CORRECT. Do NOT re-identify or change it.\n"
        "Your ONLY job: from the frames, write observation, explanation, summary, and list participants.\n\n"
        f"Video duration: {duration:.1f}s. Collision time: {accident_time:.2f}s.\n"
        f"{len(timestamps)} frames around collision: {labels}.\n\n"
        + COLLISION_MECHANISM_GUIDE
        + "\nPARTICIPANTS: list the vehicle/road-user types involved in the crash, "
        f"using only these labels: [{vocab_txt}]. Order by involvement.\n"
        "  - Classify pickup trucks, SUVs, vans, and sedans as 'car'. Reserve 'truck' "
        "for large/commercial trucks (semi, tractor-trailer, box truck).\n"
        "  - rear-end, t-bone, sideswipe, and head-on ALWAYS involve AT LEAST 2 participants; "
        "single-vehicle involves exactly 1.\n\n"
        "Below are REAL annotated examples from human experts. Match their style and granularity.\n\n"
        f"{examples_txt}\n\n"
        "=== NOW ANNOTATE THIS VIDEO ===\n"
        "KNOWN METADATA (copy semantics; do NOT restate scene/weather/lighting/type in your text "
        "unless it explains the crash):\n"
        f"  scene: {scene}\n"
        f"  weather: {fixed['weather']}\n"
        f"  lighting: {fixed['lighting']}\n"
        f"  collision_type: {collision_type}\n"
        + tbone_hint
        + "\nReply with ONLY valid JSON:\n"
        "{\n"
        '  "participants": ["car", "..."],\n'
        '  "observation": ["path/motion of each participant", "...", "contact and result"],\n'
        '  "explanation": "one sentence: core conflict mechanism",\n'
        '  "summary": "one sentence overview"\n'
        "}"
    )
