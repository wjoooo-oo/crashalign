#!/usr/bin/env bash
# 0625_Prepare data then launch two-stage training (background-safe).
# 准备数据后启动两阶段训练（适合后台运行）。

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/disk3/Jiachen/envs/py310/bin/python}"
cd "${HERE}"

echo "[$(date -Iseconds)] Starting prepare_data.py ..." | tee -a train.log
"${PY}" prepare_data.py 2>&1 | tee -a data_prep.log

echo "[$(date -Iseconds)] Data ready. Starting training ..." | tee -a train.log
bash run_train.sh 2>&1 | tee -a train.log
