"""Tracking helpers: per-track statistics and motion trails (tracking itself is ByteTrack/BoT-SORT in Ultralytics)."""
from __future__ import annotations

from collections import Counter, defaultdict, deque

import cv2
import numpy as np
import pandas as pd

from modules.detector import Detection
from utils.helpers import color_for_class


class TrackStats:
    """Accumulates one record per tracking ID across frames."""

    def __init__(self) -> None:
        self.tracks: dict[int, dict] = {}

    def update(self, dets: list[Detection], frame_idx: int) -> None:
        for d in dets:
            if d.track_id is None:
                continue
            t = self.tracks.setdefault(d.track_id, {
                "track_id": d.track_id, "object": d.label, "first_frame": frame_idx,
                "last_frame": frame_idx, "frames_seen": 0, "_conf_sum": 0.0})
            t["last_frame"] = frame_idx
            t["frames_seen"] += 1
            t["_conf_sum"] += d.conf

    def unique_counts(self) -> Counter:
        return Counter(t["object"] for t in self.tracks.values())

    def to_df(self) -> pd.DataFrame:
        cols = ["track_id", "object", "first_frame", "last_frame", "frames_seen", "avg_confidence"]
        rows = [{**{k: v for k, v in t.items() if k != "_conf_sum"},
                 "avg_confidence": round(t["_conf_sum"] / t["frames_seen"], 3)}
                for t in self.tracks.values()]
        return pd.DataFrame(rows, columns=cols).sort_values("track_id").reset_index(drop=True)


class TrailDrawer:
    """Draws the recent path (box centres) of each tracked object."""

    def __init__(self, length: int = 30) -> None:
        self.length = length
        self.history: dict[int, deque] = defaultdict(lambda: deque(maxlen=length))
        self.cls: dict[int, int] = {}

    def update(self, dets: list[Detection]) -> None:
        for d in dets:
            if d.track_id is None:
                continue
            cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
            self.history[d.track_id].append((int(cx), int(cy)))
            self.cls[d.track_id] = d.class_id

    def draw(self, frame: np.ndarray) -> np.ndarray:
        for tid, pts in self.history.items():
            if len(pts) < 2:
                continue
            color = color_for_class(self.cls.get(tid, 0))
            cv2.polylines(frame, [np.array(pts, dtype=np.int32)], False, color, 2, cv2.LINE_AA)
        return frame
