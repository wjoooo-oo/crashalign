#!/usr/bin/env bash
# 0625_Two-stage QLoRA: mixed train 1000, then gold low-LR refine.
# 两阶段 QLoRA：混合 train 1000，再用 gold 低 LR 精调。
#
# Split (5:2:3): train=1000 / val=400 / test=600
#
# Prereq:
#   python prepare_data.py
#
# Usage:
#   bash run_train.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/disk3/Jiachen/envs/py310/bin/python}"
cd "${HERE}"

# Stage 1: mixed gold+silver train (lr=1e-4).
# 阶段1：gold+silver 混合训练。
"${PY}" train_lora.py \
  --train_jsonl data/train.jsonl \
  --val_jsonl data/val.jsonl \
  --output_dir runs/stage1_mixed \
  --lr 1e-4 --epochs 1 --grad_accum 8

if [[ ! -s data/train_gold.jsonl ]]; then
  echo "ERROR: data/train_gold.jsonl missing or empty. Run prepare_data.py first."
  exit 1
fi
if [[ ! -s data/val_gold.jsonl ]]; then
  echo "Building data/val_gold.jsonl from val.jsonl (source=gold) ..."
  "${PY}" - <<'PY'
import json
from pathlib import Path
val = Path("data/val.jsonl")
out = Path("data/val_gold.jsonl")
rows = [json.loads(l) for l in val.read_text().splitlines() if l.strip()]
gold = [r for r in rows if r.get("source") == "gold"]
out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in gold) + "\n")
print(f"val_gold: {len(gold)} from {len(rows)} val")
PY
fi

# 阶段2：gold 训练 + gold val；每 epoch 评估，保留 best，早停。
"${PY}" train_lora.py \
  --train_jsonl data/train_gold.jsonl \
  --val_jsonl data/val_gold.jsonl \
  --init_adapter runs/stage1_mixed \
  --output_dir runs/stage2_gold_refine \
  --lr 2e-5 --epochs 10 --grad_accum 8 \
  --early_stopping_patience 2

FINAL_ADAPTER="runs/stage2_gold_refine"
echo "Done. Final adapter: ${FINAL_ADAPTER}"
echo "Eval: ${PY} infer.py --adapter ${FINAL_ADAPTER} --test_jsonl data/test.jsonl --out ${FINAL_ADAPTER}/preds_test.jsonl"
