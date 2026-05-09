"""
Temporal-channel input construction for the YOLOv8 detector.

A frame i is encoded as a 3-channel image whose channels are grayscale frames
i-1, i, i+1. Boundary frames replicate the nearest available neighbor so the
output is always a valid 3-channel array. This is the core idea behind our
temporal variant: the detector receives motion context inside a single forward
pass, which helps for tiny / distant drones whose strongest cue is movement.

Vanilla mode (single frame, original IR JPG read as 3-channel BGR) is also
provided so the same code path serves both experimental conditions.
"""
from pathlib import Path

import cv2
import numpy as np


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def build_temporal_stack(video_dir: Path, frame_files: list, i: int) -> np.ndarray:
    n = len(frame_files)
    a = max(0, i - 1)
    b = i
    c = min(n - 1, i + 1)
    fa = to_gray(cv2.imread(str(video_dir / frame_files[a])))
    fb = to_gray(cv2.imread(str(video_dir / frame_files[b])))
    fc = to_gray(cv2.imread(str(video_dir / frame_files[c])))
    return np.stack([fa, fb, fc], axis=-1)


def build_vanilla(video_dir: Path, frame_files: list, i: int) -> np.ndarray:
    return cv2.imread(str(video_dir / frame_files[i]))


def build_input(video_dir: Path, frame_files: list, i: int, mode: str) -> np.ndarray:
    if mode == 'temporal':
        return build_temporal_stack(video_dir, frame_files, i)
    if mode == 'vanilla':
        return build_vanilla(video_dir, frame_files, i)
    raise ValueError(f'unknown mode: {mode!r} (expected "temporal" or "vanilla")')


def list_frames(video_dir: Path) -> list:
    return sorted(
        f.name for f in video_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')
    )
