"""
Run a YOLOv8 detector + SORT tracker over a folder of Anti-UAV video sequences
and emit per-video JSON predictions in the submission format used by the
challenge baseline:

    results/<run_name>/<video>.txt = {"res": [[x, y, w, h], ...]}

A predicted bbox of [0] (one-element list) means "no drone in this frame",
matching the answerability convention used by test_siamfc.py / the eval script.

The same script handles all four experimental conditions controlled by flags:

    --mode temporal --weights <fine_tuned.pt>   # our method
    --mode vanilla  --weights <fine_tuned.pt>   # proposal baseline
    --mode vanilla  --weights yolov8n.pt        # zero-shot reference
    (SiamFC is its own baseline; see experiments/04_siamfc_baseline.sh)

Use --videos val_videos.txt to evaluate on the held-out training videos
(needed for in-paper SA numbers); omit it to predict every subdir of --src.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from src import set_seed
from src.sort import Sort
from src.temporal import build_input, list_frames


def predict_video(model, video_dir: Path, mode: str, conf: float, iou: float,
                  imgsz: int, device: str, sort_kwargs: dict, use_sort: bool) -> list:
    frame_files = list_frames(video_dir)
    n = len(frame_files)
    tracker = Sort(**sort_kwargs) if use_sort else None
    out_res = []
    for i in range(n):
        frame_in = build_input(video_dir, frame_files, i, mode=mode)
        results = model.predict(
            frame_in, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False,
        )
        r = results[0]
        if len(r.boxes) == 0:
            dets = np.empty((0, 5))
        else:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy().reshape(-1, 1)
            dets = np.concatenate([xyxy, confs], axis=1)

        if use_sort:
            tracks = tracker.update(dets)
            if len(tracks) == 0:
                out_res.append([0])
            else:
                # single-object output: keep the most confident active track
                idx = int(tracks[:, 5].argmax())
                x1, y1, x2, y2 = tracks[idx, :4]
                out_res.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
        else:
            if len(dets) == 0:
                out_res.append([0])
            else:
                idx = int(dets[:, 4].argmax())
                x1, y1, x2, y2 = dets[idx, :4]
                out_res.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    return out_res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--src', required=True,
                        help='Folder containing video subdirectories (e.g. data/track2_test or data/train)')
    parser.add_argument('--out', required=True,
                        help='Output folder for per-video JSON predictions')
    parser.add_argument('--mode', choices=('temporal', 'vanilla'), required=True)
    parser.add_argument('--videos',
                        help='Optional: path to a text file listing video subdir names to evaluate '
                             '(one per line). If omitted, every subdir of --src is processed.')
    parser.add_argument('--conf', type=float, default=0.1,
                        help='YOLO confidence threshold (low because drones are tiny in IR)')
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='')
    parser.add_argument('--no-sort', action='store_true',
                        help='Disable SORT — use the highest-confidence YOLO box per frame directly')
    parser.add_argument('--sort-max-age', type=int, default=30)
    parser.add_argument('--sort-min-hits', type=int, default=3)
    parser.add_argument('--sort-iou', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    model = YOLO(args.weights)
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.videos:
        with open(args.videos) as f:
            wanted = {line.strip() for line in f if line.strip()}
        videos = sorted(v for v in src.iterdir() if v.is_dir() and v.name in wanted)
    else:
        videos = sorted(v for v in src.iterdir() if v.is_dir())

    sort_kwargs = dict(max_age=args.sort_max_age,
                       min_hits=args.sort_min_hits,
                       iou_threshold=args.sort_iou)
    use_sort = not args.no_sort

    for v in tqdm(videos, desc='videos'):
        out_res = predict_video(
            model, v, mode=args.mode, conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, device=args.device,
            sort_kwargs=sort_kwargs, use_sort=use_sort,
        )
        (out / f'{v.name}.txt').write_text(json.dumps({'res': out_res}))


if __name__ == '__main__':
    main()
