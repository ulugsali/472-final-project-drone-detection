"""
Fine-tune YOLOv8 on the Anti-UAV thermal IR dataset.

After training finishes we record the path to best.pt at runs/<name>.path so
downstream experiment scripts can locate the weights without having to
reverse-engineer ultralytics' save-dir layout (which depends on the user's
~/Library/Application Support/Ultralytics/settings.json runs_dir + task
prefix and is not stable across versions).

Examples:
    # quick smoke test
    python -m src.train --data data/yolo_temporal/data.yaml --epochs 3 --device cpu --name smoke

    # full run on Apple Silicon
    python -m src.train --data data/yolo_temporal/data.yaml --epochs 50 --device mps --name temporal

    # bigger backbone for the moonshot
    python -m src.train --data data/yolo_temporal/data.yaml --model yolov8s.pt --epochs 50 --name temporal_s
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from src import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data.yaml from src/data_prep.py')
    parser.add_argument('--model', default='yolov8n.pt',
                        help='Pretrained YOLOv8 weights (n/s/m/l/x).')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--name', required=True,
                        help='Run name — used to disambiguate temporal vs vanilla runs')
    parser.add_argument('--device', default='',
                        help="'' = auto, 'cpu', '0' = CUDA gpu0, 'mps' = Apple Silicon")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device,
        seed=args.seed,
    )

    # ultralytics decides where to save based on its settings; record the path
    # so experiment scripts don't have to guess.
    save_dir = Path(model.trainer.save_dir)
    best_pt = save_dir / 'weights' / 'best.pt'
    if not best_pt.exists():
        raise SystemExit(f'expected best.pt at {best_pt} but file is missing')

    manifest_dir = Path('runs')
    manifest_dir.mkdir(parents=True, exist_ok=True)
    # write a relative path (relative to the project root) so the manifest is
    # portable across machines when the project is shared as a zip
    try:
        rel_path = best_pt.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        rel_path = best_pt
    (manifest_dir / f'{args.name}.path').write_text(str(rel_path) + '\n')
    print(f'\nbest weights: {best_pt}')
    print(f'recorded at:  runs/{args.name}.path')


if __name__ == '__main__':
    main()
