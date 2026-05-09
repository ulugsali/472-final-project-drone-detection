#!/usr/bin/env bash
# Generate predictions on the unlabeled CodaLab test set (data/track2_test/) for
# leaderboard submission. There are no labels here — output JSONs are zipped
# and uploaded to the challenge.
set -euo pipefail

WEIGHTS=${WEIGHTS:-$(cat runs/temporal.path 2>/dev/null || echo runs/temporal.path-MISSING)}
MODE=${MODE:-temporal}
DEVICE=${DEVICE:-mps}
OUT=${OUT:-results/submission_${MODE}}

python -m src.inference \
    --weights "$WEIGHTS" \
    --src data/track2_test \
    --out "$OUT" \
    --mode "$MODE" \
    --device "$DEVICE"

cd "$(dirname "$OUT")"
zip -r "submission_${MODE}.zip" "$(basename "$OUT")"
echo "wrote submission archive: $(pwd)/submission_${MODE}.zip"
