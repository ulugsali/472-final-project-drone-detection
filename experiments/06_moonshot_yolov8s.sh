#!/usr/bin/env bash
# Moon-shot experiment: train YOLOv8s on temporal-stacked frames with stride 10,
# validation videos used in the proposal (33 videos). Reports State Accuracy (SA) for
# comparison against vanilla and temporal baselines.
# Lower epochs for multiple test runs to find most optimal
set -euo pipefail

DEVICE=${DEVICE:-mps}
EPOCHS=${EPOCHS:-25}

python -m src.data_prep \
    --src data/train \
    --dst data/yolo_moonshot \
    --mode temporal \
    --stride 10 \
    --seed 42

python -m src.train \
    --data data/yolo_moonshot/data.yaml \
    --model src/weights/yolov8s.pt \
    --epochs "$EPOCHS" \
    --name moonshot_yolov8s \
    --device "$DEVICE" \
    --seed 42

WEIGHTS=$(cat runs/moonshot_yolov8s.path)
python -m src.inference \
    --weights "$WEIGHTS" \
    --src data/train \
    --videos data/yolo_moonshot/val_videos.txt \
    --out results/moonshot_yolov8s_sort \
    --mode temporal \
    --device "$DEVICE"

python -m src.evaluate \
    --pred results/moonshot_yolov8s_sort \
    --gt-src data/train \
    --videos data/yolo_moonshot/val_videos.txt \
    --tag moonshot_yolov8s \
    --out-csv results/moonshot_yolov8s_sort.csv