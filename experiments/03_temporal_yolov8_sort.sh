#!/usr/bin/env bash
# Reproduces the "Fine-tuned YOLOv8 + SORT (temporal input)" row — our method.
# The detector sees grayscale (t-1, t, t+1) stacked as 3 channels instead of
# the original IR-as-RGB encoding. This is the key novel contribution.
set -euo pipefail

DEVICE=${DEVICE:-mps}
EPOCHS=${EPOCHS:-50}

python -m src.data_prep \
    --src data/train \
    --dst data/yolo_temporal \
    --mode temporal \
    --stride 10 \
    --seed 42

python -m src.train \
    --data data/yolo_temporal/data.yaml \
    --model src/weights/yolov8n.pt \
    --epochs "$EPOCHS" \
    --name temporal \
    --device "$DEVICE" \
    --seed 42

WEIGHTS=$(cat runs/temporal.path)
python -m src.inference \
    --weights "$WEIGHTS" \
    --src data/train \
    --videos data/yolo_temporal/val_videos.txt \
    --out results/temporal_yolov8_sort \
    --mode temporal \
    --device "$DEVICE"

python -m src.evaluate \
    --pred results/temporal_yolov8_sort \
    --gt-src data/train \
    --videos data/yolo_temporal/val_videos.txt \
    --tag temporal_yolov8_sort \
    --out-csv results/temporal_yolov8_sort.csv
