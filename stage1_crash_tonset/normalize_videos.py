#!/usr/bin/env python3
"""Re-encode a split to a common capture format to remove source-appearance cues.

ACCIDENT clips are 960x720-1920x1080 with mixed frame rates; SO-TAD clips are
uniformly 640x480 @ 25 fps. Those low-level differences identify the source
collection, and therefore the crash label, from a single frame. This script
rewrites both pools with identical resolution, frame rate and codec settings.

Duration is deliberately left alone: it is neutralised later by the
fixed-length window crop, and truncating here would drop annotated onsets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from crashalign_paths import accident_root  # noqa: E402

# Source videos are read from the data tree. / 源视频从数据目录读取。
ROOT = accident_root()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def normalize_one(meta: dict, out_root: Path, width: int, height: int, fps: int, crf: int):
    src = ROOT / meta["path"]
    rel = Path(meta["path"])
    dst = out_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return meta, "skip"
    if not src.exists():
        return meta, "missing"
    cmd = [
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
        "-i", str(src),
        # force the same pixel grid for both pools; SAR reset avoids stretched output
        "-vf", f"scale={width}:{height}:flags=bicubic,setsar=1",
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-an",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0 or not dst.exists():
        return meta, "fail"
    return meta, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(HERE / "data_split"))
    p.add_argument("--split", required=True, choices=("train", "val", "test"))
    p.add_argument("--out_videos", default=str(HERE / "norm_videos"))
    p.add_argument("--out_data_dir", default=str(HERE / "data_split_norm"))
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_videos)
    out_data = Path(args.out_data_dir)
    out_data.mkdir(parents=True, exist_ok=True)

    counts = {}
    for kind in ("crash", "non"):
        metas = load_jsonl(data_dir / f"{args.split}_{kind}.jsonl")
        if args.limit:
            metas = metas[: args.limit]
        stats = {"ok": 0, "skip": 0, "fail": 0, "missing": 0}
        new_metas = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(normalize_one, m, out_root, args.width, args.height, args.fps, args.crf)
                for m in metas
            ]
            for f in tqdm(futs, desc=f"normalize-{args.split}-{kind}"):
                meta, status = f.result()
                stats[status] += 1
                if status in ("ok", "skip"):
                    m = dict(meta)
                    m["path"] = str(
                        (out_root / Path(meta["path"])).relative_to(ROOT)
                    )
                    m["source_path"] = meta["path"]
                    new_metas.append(m)
        with (out_data / f"{args.split}_{kind}.jsonl").open("w") as f:
            for m in new_metas:
                f.write(json.dumps(m) + "\n")
        counts[kind] = stats
        print(f"{kind}: {stats}")

    meta_path = out_data / f"normalize_meta_{args.split}.json"
    meta_path.write_text(
        json.dumps(
            {
                "split": args.split,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "codec": f"libx264 crf={args.crf} preset=veryfast yuv420p",
                "duration": "unchanged (neutralised by fixed-length window crop)",
                "counts": counts,
                "source_data_dir": str(data_dir),
            },
            indent=2,
        )
    )
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found")
    main()
