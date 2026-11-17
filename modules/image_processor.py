"""Single-image detection pipeline."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from modules.detector import Detector, annotate


def run_image_detection(detector: Detector, image_bgr: np.ndarray, conf: float, iou: float,
                        imgsz: int, classes: list[int] | None) -> dict:
    res = detector.detect(image_bgr, conf, iou, imgsz, classes)
    dets = res.detections
    return {
        "annotated": annotate(image_bgr, dets),
        "detections": dets,
        "counts": Counter(d.label for d in dets),
        "confs": [d.conf for d in dets],
        "inference_ms": res.inference_ms,
        "total_ms": res.total_ms,
        "shape": image_bgr.shape[:2],
    }


def detections_to_df(dets) -> pd.DataFrame:
    rows = [{"object": d.label, "confidence": round(d.conf, 3),
             "x1": int(d.box[0]), "y1": int(d.box[1]), "x2": int(d.box[2]), "y2": int(d.box[3])}
            for d in dets]
    return pd.DataFrame(rows, columns=["object", "confidence", "x1", "y1", "x2", "y2"])
