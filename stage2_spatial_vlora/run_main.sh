#!/usr/bin/env bash
# 0917_Train main config: frozen vision blocks + merger LoRA + LLM LoRA.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/disk3/Jiachen/envs/py310/bin/python}"
cd "${HERE}"
[[ -s data/train.jsonl ]] || "${PY}" prepare_data.py
"${PY}" train.py --config main --output_dir runs/main --epochs 10 --patience 2
