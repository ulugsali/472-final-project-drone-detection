# Data

This folder holds the Anti-UAV Track 2 dataset. The data itself is too large to
ship with the repo (~10 GB train + ~10 GB test) — download from the challenge
and place the videos here.

## Expected layout

```
data/
├── train/                    # 223 training videos with labels
│   ├── 20190925_101846_1_1/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── ...
│   │   └── IR_label.json     # {"exist": [...], "gt_rect": [[x,y,w,h], ...], ...}
│   └── ...
└── track2_test/              # 216 test videos (no labels — for CodaLab submission)
    ├── 02_6319_0000-1499/
    │   ├── 000001.jpg
    │   └── ...
    └── ...
```

## Source

4th Anti-UAV Workshop & Challenge (CVPR 2025), Track 2:
<https://codalab.lisn.upsaclay.fr/competitions/21690>

## Download

Download the dataset from Google Drive:

- Training set (223 videos, ~20 GB): [Download](https://drive.google.com/file/d/1EphVNGRofkgD0qI0hY--O37obMaT0YZK/view?usp=drive_link)
- Test set (216 videos, ~10 GB): [Download](https://drive.google.com/file/d/1LNkuiGdNG-V292WYEWejnCfCArFy7Aee/view?usp=drive_link)

Download the data from the google drive links and place the downloaded folders as `data/train/` and `data/track2_test/` respectively.

## Pretrained Weights

Download the YOLO model weights and place them under `src/weights/`:

- YOLOv8n (nano, 3.2M parameters): [Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt)
- YOLOv8s (small, 11.2M parameters): [Download](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt)
- YOLO26n (nano, 2.6M parameters): [Download](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt)


## Generated subfolders

`src/data_prep.py` writes YOLOv8-format datasets into siblings of train/, e.g.
`data/yolo_temporal/`, `data/yolo_vanilla/` and `data/yolo_moonshot/`. These are derived artifacts —
safe to delete and regenerate.
