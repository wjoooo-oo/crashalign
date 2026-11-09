#!/usr/bin/env bash
# 0914_Full spatial LoRA pipeline: prepare → train → ZS vs LoRA (separate procs).
# 完整流程：准备数据 → 训练 → 分进程评测零样本与 LoRA（避免 OOM）。
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/disk3/Jiachen/envs/py310/bin/python}"
cd "${HERE}"

"${PY}" prepare_data.py
bash run_train.sh

mkdir -p runs/eval

# Process 1: zero-shot only (no adapter).
# 进程1：仅零样本。
"${PY}" infer_eval.py --split test --out_dir runs/eval \
  2>&1 | tee runs/eval/test_zeroshot_only.log

# Process 2: spatial LoRA only (fresh GPU context).
# 进程2：仅 spatial LoRA（新进程，避免二次加载 OOM）。
"${PY}" infer_eval.py --adapter runs/spatial_qlora --split test --out_dir runs/eval \
  2>&1 | tee runs/eval/test_lora_only.log

echo "Done. See runs/eval/test_*_preds.jsonl and README results."
