"""
Convert Anti-UAV training videos into a YOLOv8-format dataset.

Two modes:
    --mode temporal : 3-channel = grayscale frames (t-1, t, t+1)   [our method]
    --mode vanilla  : 3-channel = original IR JPG read as BGR      [proposal baseline]

Train/val split is per-VIDEO, not per-frame — a video's frames are highly
correlated, so frame-level splitting would leak. Both modes share the same
seeded video split, which means temporal vs vanilla are evaluated on identical
held-out videos and results are directly comparable.

The list of validation video names is written to <dst>/val_videos.txt so that
src/inference.py and src/evaluate.py can reuse it.

Usage:
    python -m src.data_prep --mode temporal --src data/train --dst data/yolo_temporal
    python -m src.data_prep --mode vanilla  --src data/train --dst data/yolo_vanilla
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

from src import set_seed
from src.temporal import build_input, list_frames

CLASS_DRONE = 0


def bbox_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int):
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return cx, cy, w / img_w, h / img_h


def split_videos(videos: list, val_frac: float, seed: int) -> set:
    rng = random.Random(seed)
    shuffled = sorted(videos, key=lambda p: p.name)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    return {v.name for v in shuffled[:n_val]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='data/train')
    parser.add_argument('--dst', required=True,
                        help='Output dataset folder (e.g. data/yolo_temporal)')
    parser.add_argument('--mode', choices=('temporal', 'vanilla'), required=True)
    parser.add_argument('--stride', type=int, default=10,
                        help='Subsample every Nth frame (smaller = more samples, slower training)')
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42,
                        help='Controls the per-video train/val split — keep constant across modes')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate even if --dst already contains a prepared dataset')
    args = parser.parse_args()
    set_seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        raise SystemExit(f'src not found: {src.resolve()}')

    yaml_path = dst / 'data.yaml'
    val_list_path = dst / 'val_videos.txt'
    if yaml_path.exists() and val_list_path.exists() and not args.force:
        print(f'[skip] {dst} already prepared — pass --force to regenerate')
        return

    for split in ('train', 'val'):
        (dst / 'images' / split).mkdir(parents=True, exist_ok=True)
        (dst / 'labels' / split).mkdir(parents=True, exist_ok=True)

    videos = sorted(v for v in src.iterdir() if v.is_dir())
    val_names = split_videos(videos, args.val_frac, args.seed)
    print(f'{len(videos)} videos | {len(val_names)} val | {len(videos) - len(val_names)} train')
    print(f'mode={args.mode} stride={args.stride} seed={args.seed}')

    total = 0
    for v in videos:
        split = 'val' if v.name in val_names else 'train'
        label_path = v / 'IR_label.json'
        if not label_path.exists():
            print(f'  skip {v.name}: no IR_label.json')
            continue
        with open(label_path) as f:
            lab = json.load(f)
        frame_files = list_frames(v)
        n = min(len(frame_files), len(lab['exist']))

        n_written = 0
        for i in range(0, n, args.stride):
            img = build_input(v, frame_files, i, mode=args.mode)
            h, w = img.shape[:2]
            base = f'{v.name}_{i:06d}'
            cv2.imwrite(str(dst / 'images' / split / f'{base}.jpg'), img)

            lines = []
            if lab['exist'][i] == 1:
                gt = lab['gt_rect'][i]
                if len(gt) == 4 and gt[2] > 0 and gt[3] > 0:
                    cx, cy, bw, bh = bbox_to_yolo(*gt, w, h)
                    lines.append(f'{CLASS_DRONE} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
            (dst / 'labels' / split / f'{base}.txt').write_text('\n'.join(lines))
            n_written += 1
        total += n_written
        print(f'  {v.name} [{split}]: {n_written} samples')

    yaml_text = (
        f'path: {dst.resolve()}\n'
        'train: images/train\n'
        'val: images/val\n'
        'names:\n'
        '  0: drone\n'
    )
    (dst / 'data.yaml').write_text(yaml_text)
    (dst / 'val_videos.txt').write_text('\n'.join(sorted(val_names)) + '\n')
    print(f'\nwrote {total} samples')
    print(f'wrote {dst / "data.yaml"}')
    print(f'wrote {dst / "val_videos.txt"} ({len(val_names)} videos)')


if __name__ == '__main__':
    main()
