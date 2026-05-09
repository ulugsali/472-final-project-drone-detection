#!/usr/bin/env bash
# Moon-shot experiment: train YOLO26n on temporal-stacked frames with stride 10,
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
    --model src/weights/yolo26n.pt \
    --epochs "$EPOCHS" \
    --name moonshot_yolo26n \
    --device "$DEVICE" \
    --seed 42

WEIGHTS=$(cat runs/moonshot_yolo26n.path)
python -m src.inference \
    --weights "$WEIGHTS" \
    --src data/train \
    --videos data/yolo_moonshot/val_videos.txt \
    --out results/moonshot_yolo26n_sort \
    --mode temporal \
    --device "$DEVICE"

python -m src.evaluate \
    --pred results/moonshot_yolo26n_sort \
    --gt-src data/train \
    --videos data/yolo_moonshot/val_videos.txt \
    --tag moonshot_yolo26n \
    --out-csv results/moonshot_yolo26n_sort.csv