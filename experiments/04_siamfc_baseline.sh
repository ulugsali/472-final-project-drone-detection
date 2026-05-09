#!/usr/bin/env bash
# Reproduces the SiamFC reference-baseline row.
#
# Note: The official SiamFC baseline assumes the ground-truth bbox in the first
# frame, which Track 2 does NOT provide. The reported lower-bound number for
# Track 2 (SA = 0.0745) comes from the official Anti-UAV repo without a
# detector to seed initialization. We cite that number directly in the paper
# rather than re-running on our val split, because re-running it identically
# would require the original repo's eval pipeline.
#
# This script exists as a placeholder pointer to legacy/test_siamfc.py for
# anyone who wants to run SiamFC themselves on the train/val split.
set -euo pipefail

cat <<'EOF'
SiamFC baseline reference: Track 2 SA = 0.0745
  source: official Anti-UAV repo (https://github.com/ZhaoJ9014/Anti-UAV)
  cited in proposal section 4 ("Official Organizer Baseline")

Optional re-run on the held-out val split:
    1. Edit legacy/test_siamfc.py to point video_paths -> data/train and
       res_file -> data/train/<video>/IR_label.json
    2. Filter to videos in data/yolo_vanilla/val_videos.txt
    3. python legacy/test_siamfc.py
EOF
