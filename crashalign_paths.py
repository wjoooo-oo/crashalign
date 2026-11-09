"""Locate the ACCIDENT data tree used by the CrashAlign pipeline.

This GitHub package keeps the paper's training and inference code.
Videos, annotation CSVs, frame caches, and checkpoints stay outside the
repository. Point CRASHALIGN_DATA at the original working tree.

本文件只解析数据根目录。GitHub 包里是论文主流程代码；
视频、标注、帧缓存和权重不进仓库。用 CRASHALIGN_DATA 指向原来的工作目录。
"""

from __future__ import annotations

import os
from pathlib import Path

# Directory that contains this file, i.e. the code package root.
# 本文件所在目录，即代码包根目录。
PACKAGE_ROOT = Path(__file__).resolve().parent


def accident_root() -> Path:
    """Return the tree that holds videos and annotations.

    返回存放视频和标注的目录。
    Priority / 优先级:
      1. CRASHALIGN_DATA
      2. a sibling folder named accident (the layout on the development machine)
      3. this package, so missing-data errors name a real path
    """
    env = os.environ.get("CRASHALIGN_DATA", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    sibling = PACKAGE_ROOT.parent / "accident"
    if sibling.is_dir():
        return sibling.resolve()
    return PACKAGE_ROOT
