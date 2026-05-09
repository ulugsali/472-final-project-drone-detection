#!/usr/bin/env bash
# Reproduces the "Fine-tuned YOLOv8 + SORT (vanilla input)" row.
# This is the architecture committed to in the project proposal.
set -euo pipefail

DEVICE=${DEVICE:-mps}
EPOCHS=${EPOCHS:-50}

# 1) Build the vanilla single-frame dataset (one-time).
python -m src.data_prep \
    --src data/train \
    --dst data/yolo_vanilla \
    --mode vanilla \
    --stride 10 \
    --seed 42

# 2) Fine-tune YOLOv8n on it.
python -m src.train \
    --data data/yolo_vanilla/data.yaml \
    --model src/weights/yolov8n.pt \
    --epochs "$EPOCHS" \
    --name vanilla \
    --device "$DEVICE" \
    --seed 42

# 3) Run YOLO+SORT inference on the held-out validation videos.
WEIGHTS=$(cat runs/vanilla.path)
python -m src.inference \
    --weights "$WEIGHTS" \
    --src data/train \
    --videos data/yolo_vanilla/val_videos.txt \
    --out results/vanilla_yolov8_sort \
    --mode vanilla \
    --device "$DEVICE"

# 4) Score with State Accuracy.
python -m src.evaluate \
    --pred results/vanilla_yolov8_sort \
    --gt-src data/train \
    --videos data/yolo_vanilla/val_videos.txt \
    --tag vanilla_yolov8_sort \
    --out-csv results/vanilla_yolov8_sort.csv
