"""
Minimal SORT (Simple Online and Realtime Tracking) implementation.

Reference: Bewley, Ge, Ott, Ramos, Upcroft. "Simple Online and Realtime
Tracking", ICIP 2016. Original BSD-3-licensed code at
https://github.com/abewley/sort. This is a clean rewrite that adheres to the
same algorithm: per-track Kalman filter on a constant-velocity bbox model,
Hungarian assignment of detections to predicted tracks via IoU.

Detection confidence is propagated through the tracker so downstream code can
pick the best track per frame (single-object output for Anti-UAV Track 2).
"""
from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def _xyxy_to_z(bbox: np.ndarray) -> np.ndarray:
    """Convert [x1,y1,x2,y2] -> [cx, cy, scale=area, ratio=w/h] for the Kalman state."""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h) if h > 0 else 0.0
    return np.array([cx, cy, s, r]).reshape(4, 1)


def _z_to_xyxy(x: np.ndarray) -> np.ndarray:
    cx, cy, s, r = x[0], x[1], x[2], x[3]
    s = max(float(s), 1e-6)
    r = max(float(r), 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]).flatten()


def iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4) -> (N,M) matrix."""
    a = np.expand_dims(a, 1)
    b = np.expand_dims(b, 0)
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    inter = w * h
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area_a + area_b - inter + 1e-9)


class _KalmanBoxTracker:
    _next_id = 0

    def __init__(self, bbox: np.ndarray, conf: float):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        # state: [cx, cy, s, r, vx, vy, vs]
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ])
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0  # high uncertainty on initial velocity
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = _xyxy_to_z(bbox)

        _KalmanBoxTracker._next_id += 1
        self.id = _KalmanBoxTracker._next_id
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.time_since_update = 0
        self.last_conf = float(conf)

    def predict(self) -> np.ndarray:
        # avoid scale going negative under fast updates
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return _z_to_xyxy(self.kf.x[:4].copy())

    def update(self, bbox: np.ndarray, conf: float) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.last_conf = float(conf)
        self.kf.update(_xyxy_to_z(bbox))

    def state(self) -> np.ndarray:
        return _z_to_xyxy(self.kf.x[:4].copy())


def _associate(detections: np.ndarray, tracks: np.ndarray, iou_threshold: float):
    """Hungarian assignment of dets to tracks under IoU >= threshold."""
    if len(tracks) == 0 or len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.arange(len(tracks)),
        )

    iou = iou_batch(detections, tracks)
    # maximize IoU -> minimize -IoU
    row, col = linear_sum_assignment(-iou)
    matches = []
    unmatched_d = list(range(len(detections)))
    unmatched_t = list(range(len(tracks)))
    for r, c in zip(row, col):
        if iou[r, c] < iou_threshold:
            continue
        matches.append((r, c))
        unmatched_d.remove(r)
        unmatched_t.remove(c)
    return (
        np.array(matches, dtype=int).reshape(-1, 2),
        np.array(unmatched_d, dtype=int),
        np.array(unmatched_t, dtype=int),
    )


class Sort:
    """Multi-object SORT tracker. update() returns active tracks each frame."""

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list = []
        self.frame_count = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        """
        Args:
            dets: (N, 5) array of [x1, y1, x2, y2, conf]; pass np.empty((0,5)) for "no detections".
        Returns:
            (M, 6) array of [x1, y1, x2, y2, track_id, conf] for active tracks.
        """
        self.frame_count += 1
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 5))

        # 1) Kalman-predict each existing track and drop invalid ones
        predicted = np.zeros((len(self.trackers), 4))
        invalid = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                invalid.append(t)
            predicted[t] = pos
        for t in reversed(invalid):
            self.trackers.pop(t)
            predicted = np.delete(predicted, t, axis=0)

        # 2) Associate detections to predicted tracks
        matches, unmatched_d, unmatched_t = _associate(
            dets[:, :4], predicted, self.iou_threshold
        )

        # 3) Update matched tracks with their detections
        for d, t in matches:
            self.trackers[t].update(dets[d, :4], dets[d, 4])

        # 4) Spawn a new track for each unmatched detection
        for d in unmatched_d:
            self.trackers.append(_KalmanBoxTracker(dets[d, :4], dets[d, 4]))

        # 5) Collect output and prune dead tracks
        ret = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                box = trk.state()
                ret.append([box[0], box[1], box[2], box[3], trk.id, trk.last_conf])
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
        return np.array(ret) if ret else np.empty((0, 6))
