# UAV Detection and Tracking in Thermal Infrared Video

Two-stage pipeline (YOLOv8 detector + SORT tracker) for the 4th Anti-UAV
Workshop & Challenge, Track 2. Our method replaces the detector's RGB-style
3-channel input with grayscale frames stacked across time (t-1, t, t+1), which
gives the per-frame detector motion context for free — particularly useful for
tiny / distant drones where movement is the strongest cue.

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset under `data/` (see [data/README.md](data/README.md)):
We use pretrained YOLO models. Follow the instructions in [data/README.md](data/README.md) to place them in `src/weights/`


```
data/
├── train/        # 223 videos, each with frames + IR_label.json
└── track2_test/  # 216 videos, frames only (CodaLab submission set)
```

## Reproducing the main results

Each experiment in `experiments/` is a single bash script that materializes a
dataset, trains, runs inference on the held-out val split, and computes the
Anti-UAV State Accuracy (SA) metric. Random seed is fixed to 42 in every
stage so the train/val split is identical across conditions.

| # | Condition | Script |
|---|-----------|--------|
| 1 | Zero-shot YOLOv8 (no fine-tune) — measures the IR domain gap | `experiments/01_zeroshot_yolov8.sh` |
| 2 | Fine-tuned YOLOv8 + SORT, vanilla single-frame input — proposal baseline | `experiments/02_vanilla_yolov8_sort.sh` |
| 3 | Fine-tuned YOLOv8 + SORT, **temporal-channel input (ours)** | `experiments/03_temporal_yolov8_sort.sh` |
| 4 | SiamFC reference (cited from official repo: SA = 0.0745) | `experiments/04_siamfc_baseline.sh` |
| 5 | Generate CodaLab test-set submission ZIP from a trained model | `experiments/05_test_set_submission.sh` |
| 6 | Moon-shot: YOLOv8s (larger yolo model) with temporal input | `experiments/06_moonshot_yolov8s.sh` |
| 7 | Moon-shot: YOLO26n (newer yolo model with STAL) with temporal input | `experiments/07_moonshot_yolo26n.sh` |

Run them with the device that fits your hardware:

```bash
DEVICE=mps EPOCHS=50 bash experiments/03_temporal_yolov8_sort.sh   # Apple Silicon
DEVICE=0   EPOCHS=50 bash experiments/03_temporal_yolov8_sort.sh   # NVIDIA GPU
DEVICE=cpu EPOCHS=3  bash experiments/03_temporal_yolov8_sort.sh   # CPU smoke test
```

The 2-vs-3 comparison is the central diagnostic experiment: same data, same
training recipe, only the input encoding differs.

**Main result:** Row 3 (temporal-channel input) achieves 0.527 SA, a 7× improvement over SiamFC.

**Note on evaluation:** The official test set labels are not publicly available. To obtain a test set State Accuracy score, predictions must be uploaded to the CodaLab challenge server. Therefore, all reported SA scores in this repository are computed on a held-out validation split of 33 videos from the training set. This split is deterministic (seed=42) and consistent across all experiments.


## Hardware and runtime

| Stage | M2 MacBook Pro (mps) | NVIDIA T4 / A100 |
|-------|----------------------|------------------|
| `data_prep.py` (stride=10) | ~25 min  | ~25 min (IO bound) |
| `train.py` (yolov8n, 50 epochs) | ~8 hr | ~1 hr |
| `inference.py` on 34 val videos | ~30 min | ~5 min |
| `evaluate.py` | 1 second | 1 second |

For a same-day smoke test, set `--epochs 3 --stride 30 --device cpu`.

## Layout

```
.
├── README.md                  # this file
├── requirements.txt
├── data/                      # dataset + generated YOLO subfolders (gitignored)
├── src/                       # source code (see module-level docstrings)
│   ├── temporal.py            # temporal-stack input builder
│   ├── data_prep.py           # train videos -> YOLOv8 dataset
│   ├── train.py               # YOLOv8 fine-tuning
│   ├── sort.py                # vendored SORT tracker (BSD, attributed in file)
│   ├── inference.py           # YOLOv8 + SORT predictor
│   ├── evaluate.py            # State Accuracy metric on val split
│   └── baselines/
│       ├── siamfc.py          # SiamFC tracker (challenge baseline)
│       └── siamfc_model.pth   # SiamFC pretrained weights
├── experiments/               # one shell script per row of the results table
├── results/                   # predictions, CSVs, model checkpoints (gitignored)
└── legacy/                    # original challenge baseline (siamfc + yolov5),
                               # kept for reference and reproducibility audits
```

## Reproducibility

- Seed = 42 everywhere (numpy, random, PyTorch, ultralytics, video split).
- Per-video train/val split is deterministic across modes — temporal and
  vanilla evaluate on identical videos.
- `data_prep.py` writes `val_videos.txt` so downstream scripts pick up the
  same split without re-deriving it.

## Code provenance

- `src/sort.py` is a clean rewrite of [Bewley et al., SORT](https://github.com/abewley/sort) (BSD-3).
- `src/baselines/siamfc.py` is the challenge organizers' baseline tracker,
  unmodified from <https://github.com/ZhaoJ9014/Anti-UAV>.
