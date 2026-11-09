#!/usr/bin/env bash
# 0917_Train vision_lora ablation: main + last-4 vision block LoRA.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/disk3/Jiachen/envs/py310/bin/python}"
cd "${HERE}"
[[ -s data/train.jsonl ]] || "${PY}" prepare_data.py
"${PY}" train.py --config vision_lora --output_dir runs/vision_lora --epochs 10 --patience 2
