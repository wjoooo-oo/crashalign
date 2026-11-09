#!/usr/bin/env python3
"""0925_Partially fine-tune the VideoMAE window encoder on ACCIDENT (stage 1 of 2).

在 ACCIDENT 上部分微调 VideoMAE 窗口编码器（两步法的第一步）。

Why partial / 为什么只解冻顶部:
    A full fine-tune from videomae-base on matched clips collapsed (test AUROC 0.504,
    `results/accident_matched_baselines/videomae_accident`). Here the lower blocks stay
    frozen and only the top K blocks adapt, starting from the current SO-TAD encoder
    (`vit/checkpoints_temporal_clean`), so K = 0 is exactly today's frozen features.
    从 videomae-base 全量微调已失败（测试 AUROC 0.504）。这里冻结底层、只解冻顶部 K 个 block，
    并从当前 SO-TAD 编码器出发，K = 0 即当前的冻结特征。

Supervision / 监督信号 (window level, train split only / 仅用训练集):
    positive: window centre within ±1 s of onset (the 2 s window contains the impact)
              正样本：窗口中心距 onset ≤ 1 s（2 s 窗口内包含碰撞时刻）
    negative: window centre in [onset − 9 s, onset − 3 s] (clean pre-crash driving,
              same video, the hardest crash-adjacent part)
              负样本：窗口中心位于 [onset − 9 s, onset − 3 s]（同视频、碰撞前最难的干净驾驶段）
    Windows in between or after the crash are ignored (ambiguous for this task).
    介于两者之间及碰撞之后的窗口不参与训练（对本任务标签含糊）。
    This uses all ~1417 train videos, far more than the 394 gap-2s clip pairs.
    可使用全部约 1417 个训练视频，远多于 394 个 gap-2s 片段对。

Frames match 0728_extract_accident_full.py (same window grid, 16 frames by linspace);
resizing uses an OpenCV equivalent of VideoMAEImageProcessor on a one-off frame cache,
so the saved encoder plugs straight into the extractor for stage 2.
帧采样与 0728_extract_accident_full.py 一致；缩放用 VideoMAEImageProcessor 的 OpenCV 等价实现，
并预先解码到帧缓存。保存的编码器可直接用于第二步重新提取。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from crashalign_paths import accident_root  # noqa: E402

# Encoder init and source videos are outside this package.
# 编码器初始权重和源视频不在本代码包内。
ROOT = accident_root()
DEFAULT_INIT = ROOT / "vit/checkpoints_temporal_clean/best_model"
CACHE = HERE / "cache_emb_accident_full"
WINDOW_SEC = 2.0
FRAMES_PER_WINDOW = 16
POS_HALF_SEC = 1.0  # |centre − onset| <= 1 s → positive / 正样本范围
NEG_MARGIN_SEC = 3.0  # centre <= onset − 3 s → negative / 负样本边界
# Negatives are limited to the 6 s of clean driving right before that boundary: these
# are the hard, crash-adjacent windows, and the limit bounds the frame cache size.
# 负样本只取该边界之前 6 s 的干净驾驶：这些是紧邻碰撞的难负样本，同时限制帧缓存大小。
NEG_EARLIEST_SEC = 9.0  # centre >= onset − 9 s
FRAME_CACHE = HERE / "cache_frames_0925_window_ft"


# ---------------------------------------------------------------------------
# Window index / 窗口索引
# ---------------------------------------------------------------------------


def build_window_index(split: str, cache_root: Path = CACHE) -> list[dict]:
    """List positive and negative window centres for every video in a split.

    The window grid is read from the existing cache, so centres are exactly the
    ones the extractor uses; fps comes from the video to convert back to frames.
    `cache_root` may be a symlink tree of a re-split (0926 benchmark); the frame
    cache is keyed by the physical split directory the link points to, so a re-split
    reuses the frames already decoded.
    为 split 中每个视频列出正/负窗口中心。窗口网格取自现有缓存，与提取脚本完全一致；
    fps 从视频读取，用于换算回帧号。`cache_root` 可以是重划分的软链接目录（0926 benchmark）；
    帧缓存以链接指向的物理 split 目录为键，因此重划分可复用已解码的帧。
    """
    videos = []
    for npz in sorted((cache_root / split).glob("*.npz")):
        physical_split = npz.resolve().parent.name
        with np.load(npz, allow_pickle=True) as z:
            times = z["times"].astype(np.float64)
            onset = float(z["t_gt"])
            path = str(z["path"])
        cap = cv2.VideoCapture(str(ROOT / path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()
        pos = [float(t) for t in times if abs(t - onset) <= POS_HALF_SEC]
        neg = [float(t) for t in times if onset - NEG_EARLIEST_SEC <= t <= onset - NEG_MARGIN_SEC]
        if pos:
            videos.append({"video_id": npz.stem, "split": physical_split, "path": path, "fps": fps,
                           "onset": onset, "pos": pos, "neg": neg})
    return videos


def sample_train_items(videos: list[dict], n_items: int, rng: random.Random) -> list[tuple]:
    """Balanced epoch list: half positive, half negative windows.

    Videos are drawn uniformly first, then a window inside the video, so long videos
    do not dominate. 平衡采样：一半正、一半负；先均匀选视频再选窗口，避免长视频主导。
    """
    with_neg = [v for v in videos if v["neg"]]
    items = []
    for i in range(n_items):
        if i % 2 == 0:
            v = rng.choice(videos)
            items.append((v, rng.choice(v["pos"]), 1))
        else:
            v = rng.choice(with_neg)
            items.append((v, rng.choice(v["neg"]), 0))
    rng.shuffle(items)
    return items


def val_items(videos: list[dict]) -> list[tuple]:
    """Deterministic val windows per video: nearest-to-onset positive, the latest
    clean negative (hardest, closest to the crash) and one hashed earlier negative.

    验证集每个视频固定取：最接近 onset 的正窗口、最晚的干净负窗口（最难，离碰撞最近）、
    以及一个按 hash 选的较早负窗口。
    """
    items = []
    for v in videos:
        items.append((v, min(v["pos"], key=lambda t: abs(t - v["onset"])), 1))
        if v["neg"]:
            items.append((v, v["neg"][-1], 0))
            h = int(hashlib.md5(v["video_id"].encode()).hexdigest(), 16)
            items.append((v, v["neg"][h % len(v["neg"])], 0))
    return items


# ---------------------------------------------------------------------------
# Frame loading / 读帧
# ---------------------------------------------------------------------------


IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


RESIZE_MODES = ("crop", "squash")


def frame_cache_dir(resize: str) -> Path:
    """One frame cache per resize mode. / 每种缩放方式一个帧缓存目录。"""
    return FRAME_CACHE if resize == "crop" else FRAME_CACHE.with_name(FRAME_CACHE.name + f"_{resize}")


def resize_frame(frame_rgb: np.ndarray, resize: str, size: int = 224) -> np.ndarray:
    """OpenCV version of the VideoMAEImageProcessor geometry, output kept uint8 for caching.

    crop  : shortest edge → 224 then centre crop (the processor default). On 16:9 video
            this keeps only the middle 56% of the width; 13.6% of ACCIDENT crash centres
            fall outside it, and those crashes are nearly undetectable (test AUROC ≈ 0.5).
    squash: resize the whole frame to 224×224 (aspect distorted, nothing cut off).
    Crop-mode embeddings match the HF processor to cosine ≥ 0.9998 (checked on train windows).

    VideoMAEImageProcessor 几何步骤的 OpenCV 实现，输出 uint8 便于缓存。
    crop  ：短边缩放到 224 后中心裁剪（处理器默认）。16:9 视频只保留中间 56% 宽度，
            13.6% 的碰撞中心落在其外，这些碰撞几乎无法检出（测试 AUROC 约 0.5）。
    squash：整帧缩放到 224×224（纵横比变形，但不裁掉任何内容）。
    """
    if resize == "squash":
        return cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_AREA)
    h, w = frame_rgb.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = max(size, int(w * scale)), max(size, int(h * scale))
    resized = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    top, left = (new_h - size) // 2, (new_w - size) // 2
    return resized[top : top + size, left : left + size]


def normalize(frames_uint8: np.ndarray) -> torch.Tensor:
    """[T,H,W,C] uint8 → [T,C,H,W] float with ImageNet normalisation. / 归一化为模型输入。"""
    x = frames_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(((x - IMAGE_MEAN) / IMAGE_STD).transpose(0, 3, 1, 2).copy())


def window_frame_ids(fps: float, center_sec: float) -> np.ndarray:
    """The 16 frame indices of a window, identical to window_specs + linspace.

    一个窗口的 16 个帧号，与 window_specs + linspace 完全一致。
    """
    half = max(1, int(fps * WINDOW_SEC / 2))
    center = int(round(center_sec * fps))
    start = center - half
    return np.linspace(start, max(start, center + half - 1), FRAMES_PER_WINDOW).astype(int)


def cache_video_frames(v: dict, resize: str) -> None:
    """Decode the frame range covering all of a video's windows once and store it.

    Decoding a 2 s window from 1080p H.264 needs a keyframe seek plus ~40 frames, which
    made training CPU-bound (~5 s/step). Each video is decoded once instead.
    将覆盖该视频所有训练窗口的帧段解码一次并存盘。逐窗口 seek 解码 1080p 视频使训练受 CPU 限制
    （约 5 s/步），改为每个视频只解码一次。
    """
    out = frame_cache_dir(resize) / v["split"] / f"{v['video_id']}.npy"
    if out.exists():
        return
    ids = np.concatenate([window_frame_ids(v["fps"], c) for c in v["pos"] + v["neg"]])
    first, last = max(0, int(ids.min())), int(ids.max())
    cap = cv2.VideoCapture(str(ROOT / v["path"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    frames = []
    for _ in range(first, last + 1):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(resize_frame(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), resize))
    cap.release()
    arr = np.stack(frames) if frames else np.zeros((1, 224, 224, 3), dtype=np.uint8)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    tmp.rename(out)
    (out.with_suffix(".json")).write_text(json.dumps({"first_frame": first}))


def build_frame_cache(videos: list[dict], workers: int, resize: str) -> None:
    """Fill the frame cache in parallel (skips videos already cached). / 并行构建帧缓存。"""
    from functools import partial
    from multiprocessing import Pool

    cache = frame_cache_dir(resize)
    todo = [v for v in videos if not (cache / v["split"] / f"{v['video_id']}.npy").exists()]
    if todo:
        with Pool(workers, initializer=_worker_init, initargs=(0,)) as pool:
            job = partial(cache_video_frames, resize=resize)
            for i, _ in enumerate(pool.imap_unordered(job, todo, chunksize=4), 1):
                if i % 200 == 0:
                    print(f"  frame cache {i}/{len(todo)}", flush=True)


def _worker_init(_worker_id: int) -> None:
    """One thread per worker; many workers × many threads oversubscribes the CPU.

    每个 worker 只用一个线程，避免多 worker × 多线程抢占 CPU。
    """
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


class WindowDataset(Dataset):
    """Read one window's 16 frames from the frame cache per item. / 每个 item 从帧缓存读取一个窗口。"""

    def __init__(self, items: list[tuple], train: bool, seed: int, cache_dir: Path):
        self.items = items
        self.train = train
        self.seed = seed
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        v, center, label = self.items[i]
        path = self.cache_dir / v["split"] / f"{v['video_id']}.npy"
        frames = np.load(path, mmap_mode="r")
        first = json.loads(path.with_suffix(".json").read_text())["first_frame"]
        # Guard against a decode that ended a few frames early. / 防止末尾解码少几帧时越界。
        local = np.clip(window_frame_ids(v["fps"], center) - first, 0, len(frames) - 1)
        pixel_values = normalize(np.asarray(frames[local]))
        # Horizontal flip keeps traffic semantics and doubles appearance variety.
        # 水平翻转不改变交通语义，可增加外观多样性。
        if self.train and random.Random(self.seed * 1_000_003 + i).random() < 0.5:
            pixel_values = pixel_values.flip(-1)
        return {"pixel_values": pixel_values, "label": torch.tensor(float(label)),
                "video_id": v["video_id"], "role": "pos" if label else "neg"}


# ---------------------------------------------------------------------------
# Model / 模型
# ---------------------------------------------------------------------------


class PartialVideoMAE(nn.Module):
    """VideoMAE with the bottom blocks frozen and the top `n_unfrozen` blocks trainable.

    The frozen part runs under no_grad, so memory and compute for backprop only cover
    the top blocks. Pooling matches FrozenVideoMAE ("auto" = token 0 for this checkpoint).
    底层 block 冻结并在 no_grad 下前向，反向传播只覆盖顶部 block，节省显存与计算。
    池化方式与 FrozenVideoMAE 一致（该 checkpoint 下 "auto" 即第 0 个 token）。
    """

    def __init__(self, init_dir: str, n_unfrozen: int, pooling: str, grad_checkpoint: bool = False):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint
        self.base = VideoMAEForVideoClassification.from_pretrained(init_dir, num_labels=1)
        vm = self.base.videomae
        if self.base.fc_norm is not None:
            raise ValueError("expects a use_mean_pooling=False checkpoint / 需要非均值池化的 checkpoint")
        self.pooling = pooling
        layers = vm.encoder.layer
        self.n_frozen = len(layers) - n_unfrozen
        for p in self.base.parameters():
            p.requires_grad = False
        for layer in layers[self.n_frozen:]:
            for p in layer.parameters():
                p.requires_grad = True
        if n_unfrozen > 0:
            for p in vm.layernorm.parameters():
                p.requires_grad = True
        # Fresh linear head: the old classifier was trained on token-0 onset scores.
        # 新建线性头：原分类头是在 token-0 的 onset 回归上训练的。
        self.head = nn.Linear(self.base.config.hidden_size, 1)

    def backbone_params(self):
        return [p for p in self.base.parameters() if p.requires_grad]

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vm = self.base.videomae
        layers = vm.encoder.layer
        with torch.no_grad():
            h = vm.embeddings(pixel_values, None)
            for layer in layers[: self.n_frozen]:
                h = layer(h)[0]
        for layer in layers[self.n_frozen:]:
            if self.grad_checkpoint and self.training:
                # Recompute activations in backward instead of storing them (needed for K >= 8).
                # 反向时重算激活而不保存，用计算换显存（K >= 8 时需要）。
                h = torch.utils.checkpoint.checkpoint(lambda x, m=layer: m(x)[0], h, use_reentrant=False)
            else:
                h = layer(h)[0]
        h = vm.layernorm(h)
        return h.mean(1) if self.pooling == "mean" else h[:, 0]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(pixel_values)).squeeze(-1)


# ---------------------------------------------------------------------------
# Train / eval / 训练与评估
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model: PartialVideoMAE, loader: DataLoader, device) -> dict:
    """Window AUROC plus same-video pair accuracy (positive beats latest negative).

    窗口级 AUROC，以及同视频成对准确率（正窗口得分高于最晚负窗口的比例）。
    """
    model.eval()
    scores, labels, vids, roles = [], [], [], []
    for b in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s = torch.sigmoid(model(b["pixel_values"].to(device)).float())
        scores += s.cpu().tolist()
        labels += b["label"].tolist()
        vids += b["video_id"]
        roles += b["role"]
    pos, hard_neg = {}, {}
    for s, v, r in zip(scores, vids, roles):
        if r == "pos":
            pos[v] = s
        elif v not in hard_neg:  # first negative per video is the latest clean one
            hard_neg[v] = s
    paired = [v for v in pos if v in hard_neg]
    return {
        "window_auroc": float(roc_auc_score(labels, scores)),
        "pair_acc": float(np.mean([pos[v] > hard_neg[v] for v in paired])),
        "n_windows": len(scores),
        "n_pairs": len(paired),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--init", default=str(DEFAULT_INIT))
    p.add_argument("--n_unfrozen", type=int, default=2, help="top Transformer blocks to train / 解冻的顶部 block 数")
    p.add_argument("--pooling", choices=("auto", "mean"), default="mean")
    p.add_argument("--grad_checkpoint", action="store_true",
                   help="gradient checkpointing on trainable blocks (memory) / 可训练 block 使用梯度检查点")
    p.add_argument("--resize", choices=RESIZE_MODES, default="crop",
                   help="crop = processor default; squash = full frame to 224x224 / 缩放方式")
    p.add_argument("--lr_backbone", type=float, default=1e-5)
    p.add_argument("--lr_head", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--steps_per_epoch", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=24)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cache_root", type=Path, default=CACHE,
                   help="per-split npz tree defining train/val videos / 定义 train/val 视频的 npz 目录")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "0925_train.log"

    def log(msg: str) -> None:
        print(msg, flush=True)
        with log_path.open("a") as f:
            f.write(msg + "\n")

    # The saved processor encodes the resize mode, so the extractor reproduces it
    # automatically when loading best_model. / 保存的处理器记录缩放方式，提取脚本加载 best_model 时自动复现。
    if args.resize == "squash":
        processor = VideoMAEImageProcessor.from_pretrained(
            args.init, size={"height": 224, "width": 224}, do_center_crop=False)
    else:
        processor = VideoMAEImageProcessor.from_pretrained(args.init)
    cache_dir = frame_cache_dir(args.resize)
    train_videos = build_window_index("train", args.cache_root)
    val_videos = build_window_index("val", args.cache_root)
    log(f"train videos={len(train_videos)} (with neg {sum(bool(v['neg']) for v in train_videos)}) "
        f"val videos={len(val_videos)}")
    build_frame_cache(train_videos + val_videos, args.num_workers, args.resize)

    model = PartialVideoMAE(args.init, args.n_unfrozen, args.pooling, args.grad_checkpoint).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"n_unfrozen={args.n_unfrozen} pooling={args.pooling} trainable params={n_train:,}")

    param_groups = [{"params": list(model.head.parameters()), "lr": args.lr_head}]
    if args.n_unfrozen > 0:
        param_groups.append({"params": model.backbone_params(), "lr": args.lr_backbone})
    opt = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    total_steps = args.epochs * args.steps_per_epoch
    warmup = max(1, int(0.05 * total_steps))

    # Linear warmup then cosine decay, applied to every group's base lr.
    # 先线性预热再余弦衰减，作用于每个参数组的基础学习率。
    def lr_factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    loss_fn = nn.BCEWithLogitsLoss()
    val_loader = DataLoader(WindowDataset(val_items(val_videos), False, args.seed, cache_dir),
                            batch_size=16, num_workers=args.num_workers, worker_init_fn=_worker_init)
    (out / "0925_config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    best, bad, history, t0 = -1.0, 0, [], time.time()
    rng = random.Random(args.seed)
    for ep in range(1, args.epochs + 1):
        items = sample_train_items(train_videos, args.steps_per_epoch * args.batch_size, rng)
        loader = DataLoader(WindowDataset(items, True, args.seed + ep, cache_dir), batch_size=args.batch_size,
                            num_workers=args.num_workers, drop_last=True, worker_init_fn=_worker_init,
                            prefetch_factor=4)
        model.train()
        running = 0.0
        for step, b in enumerate(loader, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(b["pixel_values"].to(device, non_blocking=True))
            loss = loss_fn(logits.float(), b["label"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            running += float(loss.item())
            if step % 50 == 0:
                log(f"  ep{ep} step {step}/{args.steps_per_epoch} loss={running / step:.4f} "
                    f"{(time.time() - t0) / 60:.1f}m")
        vm = evaluate(model, val_loader, device)
        rec = {"epoch": ep, "train_loss": running / max(1, step), **vm}
        history.append(rec)
        (out / "0925_history.json").write_text(json.dumps(history, indent=2))
        log(f"[epoch {ep}] loss={rec['train_loss']:.4f} val window AUROC={vm['window_auroc']:.4f} "
            f"pair_acc={vm['pair_acc']:.4f} ({vm['n_pairs']} pairs) {(time.time() - t0) / 60:.1f}m")
        if vm["window_auroc"] > best + 1e-4:
            best, bad = vm["window_auroc"], 0
            # Saved in HF format so FrozenVideoMAE / the extractor can load it directly.
            # 以 HF 格式保存，FrozenVideoMAE 与提取脚本可直接加载。
            model.base.save_pretrained(out / "best_model")
            processor.save_pretrained(out / "best_model")
            torch.save(model.head.state_dict(), out / "best_head.pt")
            log(f"  saved best (val window AUROC {best:.4f})")
        else:
            bad += 1
            if bad >= args.patience:
                log(f"early stop at epoch {ep}")
                break
    log(f"done best val window AUROC={best:.4f}")


if __name__ == "__main__":
    main()
