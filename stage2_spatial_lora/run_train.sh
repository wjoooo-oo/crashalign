#!/usr/bin/env bash
# 0914_Train spatial-grounding QLoRA (vision frozen; separate from crash LoRA).
# 训练空间定位 QLoRA（冻结视觉；与事故理解 LoRA 分离）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/disk3/Jiachen/envs/py310/bin/python}"
cd "${HERE}"

if [[ ! -s data/train.jsonl ]]; then
  echo "Preparing data..."
  "${PY}" prepare_data.py
fi

"${PY}" train_lora.py \
  --train_jsonl data/train.jsonl \
  --val_jsonl data/val.jsonl \
  --output_dir runs/spatial_qlora \
  --lr 1e-4 \
  --epochs 8 \
  --grad_accum 8 \
  --early_stopping_patience 2 \
  --max_pixels_factor 256

echo "Done. Adapter at runs/spatial_qlora"
