#!/usr/bin/env bash
# Reproduces the "Zero-shot YOLOv8" row of the comparison table.
#
# No fine-tuning — runs the COCO-pretrained YOLOv8n on the held-out val split
# in vanilla mode. This measures the domain gap between RGB pretraining and
# the thermal IR target domain.
set -euo pipefail

VIDEOS=${VIDEOS:-data/yolo_vanilla/val_videos.txt}
DEVICE=${DEVICE:-cpu}

python -m src.inference \
    --weights src/weights/yolov8n.pt \
    --src data/train \
    --videos "$VIDEOS" \
    --out results/zeroshot_yolov8 \
    --mode vanilla \
    --no-sort \
    --device "$DEVICE"

python -m src.evaluate \
    --pred results/zeroshot_yolov8 \
    --gt-src data/train \
    --videos "$VIDEOS" \
    --tag zeroshot_yolov8 \
    --out-csv results/zeroshot_yolov8.csv
