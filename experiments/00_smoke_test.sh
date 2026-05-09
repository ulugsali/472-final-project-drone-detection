#!/usr/bin/env bash
# End-to-end smoke test on a small subset.
#
# Validates that data prep -> train -> inference -> evaluate all work and that
# evaluate.py prints a State Accuracy number. Ideal first run before
# committing to a full DEVICE=mps EPOCHS=50 training session.
#
# Defaults: stride=50 (~5k samples), 2 epochs, CPU.
set -euo pipefail

DEVICE=${DEVICE:-cpu}
EPOCHS=${EPOCHS:-2}
STRIDE=${STRIDE:-50}
DST=${DST:-data/yolo_temporal_smoke}
NAME=${NAME:-temporal_smoke}
BATCH=${BATCH:-8}

python -m src.data_prep \
    --src data/train --dst "$DST" --mode temporal --stride "$STRIDE" --seed 42

python -m src.train \
    --data "$DST/data.yaml" --model src/weights/yolov8n.pt --epochs "$EPOCHS" \
    --name "$NAME" --device "$DEVICE" --batch "$BATCH" --seed 42

WEIGHTS=$(cat "runs/$NAME.path")
python -m src.inference \
    --weights "$WEIGHTS" \
    --src data/train --videos "$DST/val_videos.txt" \
    --out "results/$NAME" --mode temporal --device "$DEVICE"

python -m src.evaluate \
    --pred "results/$NAME" --gt-src data/train \
    --videos "$DST/val_videos.txt" --tag "$NAME"
