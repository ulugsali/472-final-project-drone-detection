"""
Compute the Anti-UAV State Accuracy (SA) metric on a held-out validation split.

SA per sequence (matches the challenge eval script):

    Score = mean(M_t) - 0.2 * mean(P_t)^0.3

with eps = 1e-5,
    M_t = 1[pred_t = empty]                 if exist_t = 0
        = IoU(pred_t, gt_t)                 if exist_t = 1 and pred_t != empty
        = 0                                 if exist_t = 1 and pred_t = empty
    P_t = 0 if exist_t = 1 and IoU > eps    (correct detection)
        = 1 if exist_t = 1 and IoU <= eps   (missed detection)

Reads predictions written by src/inference.py and ground-truth IR_label.json
from the original training videos. Reports per-video SA, overall mean SA, and
optionally the mean / std across multiple runs (rubric 1C asks for statistical
context).

Usage:
    python -m src.evaluate
        --pred results/temporal_val
        --gt-src data/train
        --videos data/yolo_temporal/val_videos.txt
        --label IR_label.json
        --tag temporal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EPS = 1e-5


def iou(b1, b2) -> float:
    b1 = [float(x) for x in b1]
    b2 = [float(x) for x in b2]
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    ox1, oy1 = max(ax1, bx1), max(ay1, by1)
    ox2, oy2 = min(ax2, bx2), min(ay2, by2)
    if ox2 - ox1 <= 0 or oy2 - oy1 <= 0:
        return 0.0
    inter = (ox2 - ox1) * (oy2 - oy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def is_empty(pred) -> bool:
    return (len(pred) == 0) or (len(pred) == 1 and pred[0] == 0)


def state_accuracy(out_res: list, label: dict) -> float:
    measure = []
    penalty = []
    for pred, gt, exist in zip(out_res, label['gt_rect'], label['exist']):
        if not exist:
            measure.append(1.0 if is_empty(pred) else 0.0)
        else:
            if is_empty(pred):
                measure.append(0.0)
                penalty.append(1)
            else:
                ov = iou(pred, gt)
                measure.append(ov)
                penalty.append(0 if ov > EPS else 1)
    m = np.mean(measure) if measure else 0.0
    p = np.mean(penalty) if penalty else 0.0
    return float(m - 0.2 * (p ** 0.3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', required=True, help='Folder with <video>.txt prediction JSONs')
    parser.add_argument('--gt-src', default='data/train',
                        help='Folder containing the original video subdirs with IR_label.json files')
    parser.add_argument('--videos',
                        help='Optional text file listing video names to evaluate (one per line). '
                             'If omitted, every prediction file in --pred is evaluated.')
    parser.add_argument('--label', default='IR_label.json',
                        help='Label filename inside each video folder')
    parser.add_argument('--tag', default='run', help='Label printed alongside the score')
    parser.add_argument('--out-csv', help='Optional CSV path for per-video scores')
    args = parser.parse_args()

    pred_dir = Path(args.pred)
    gt_root = Path(args.gt_src)
    if args.videos:
        with open(args.videos) as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = sorted(p.stem for p in pred_dir.glob('*.txt'))

    rows = []
    scores = []
    missing_gt = []
    for name in names:
        pred_path = pred_dir / f'{name}.txt'
        gt_path = gt_root / name / args.label
        if not pred_path.exists():
            print(f'  skip {name}: no prediction')
            continue
        if not gt_path.exists():
            missing_gt.append(name)
            continue
        with open(pred_path) as f:
            out_res = json.load(f)['res']
        with open(gt_path) as f:
            lab = json.load(f)
        score = state_accuracy(out_res, lab)
        scores.append(score)
        rows.append((name, score))
        print(f'  {name:40s} SA={score:+.4f}')

    if missing_gt:
        print(f'\n[warn] {len(missing_gt)} videos had predictions but no ground-truth label '
              f'(expected for the public test set)')

    if scores:
        arr = np.array(scores)
        median_val = np.median(arr)
        print(f'\n[{args.tag}] N={len(arr)}  mean SA = {arr.mean():+.4f}  '
              f'std = {arr.std():.4f}  min = {arr.min():+.4f}  max = {arr.max():+.4f}  '
              f'median = {median_val:+.4f}')
    else:
        print(f'\n[{args.tag}] no scored videos (no ground truth available)')

    if args.out_csv and rows:
        with open(args.out_csv, 'w') as f:
            f.write('video,SA\n')
            for name, score in rows:
                f.write(f'{name},{score:.6f}\n')
        print(f'wrote per-video scores -> {args.out_csv}')


if __name__ == '__main__':
    main()
