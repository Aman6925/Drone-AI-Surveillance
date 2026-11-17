"""YOLO wrapper: model loading, device selection (GPU with CPU fallback), detection, tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from config import settings
from utils.helpers import InferenceError, ModelLoadError, color_for_class


@dataclass
class Detection:
    label: str
    class_id: int
    conf: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 (pixels)
    track_id: int | None = None


@dataclass
class FrameResult:
    detections: list[Detection]
    inference_ms: float  # model forward pass only (Ultralytics-reported)
    total_ms: float      # wall-clock incl. pre/post-processing


def resolve_device(pref: str = "auto") -> tuple[str, str]:
    """Return (device, note). Falls back to CPU whenever CUDA is unavailable or broken."""
    if pref == "cpu":
        return "cpu", "CPU selected manually."
    try:
        import torch

        if torch.cuda.is_available():
            torch.zeros(1, device="cuda:0")  # verify CUDA really works
            return "cuda:0", f"GPU: {torch.cuda.get_device_name(0)}"
        return "cpu", "CUDA not available - using CPU fallback."
    except Exception as exc:  # noqa: BLE001
        return "cpu", f"CUDA check failed ({exc.__class__.__name__}) - using CPU fallback."


class Detector:
    def __init__(self, model_name: str = "yolov8n.pt", device_pref: str = "auto"):
        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                "Ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.model_name = model_name
        self.device, self.device_note = resolve_device(device_pref)
        weights = settings.MODEL_DIR / model_name
        try:
            self.model = YOLO(str(weights))  # auto-downloads into models/ if missing
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Could not load '{model_name}': {exc}\n\n"
                f"If you are offline, download it from "
                f"https://github.com/ultralytics/assets/releases and place it in: {settings.MODEL_DIR}"
            ) from exc
        self.names: dict[int, str] = dict(self.model.names)
        self._warmup()

    # ------------------------------------------------------------------ internals
    def _kwargs(self, conf, iou, imgsz, classes) -> dict:
        kw = dict(conf=conf, iou=iou, imgsz=imgsz, classes=classes or None,
                  device=self.device, verbose=False)
        if self.device.startswith("cuda"):  # FP16 on GPU; argument was renamed in newer Ultralytics
            from ultralytics.cfg import DEFAULT_CFG_DICT
            if "quantize" in DEFAULT_CFG_DICT:
                kw["quantize"] = 16
            else:
                kw["half"] = True
        return kw

    def _warmup(self) -> None:
        blank = np.zeros((320, 320, 3), dtype=np.uint8)
        try:
            self.model.predict(blank, **self._kwargs(0.25, 0.45, 320, None))
        except Exception as exc:  # noqa: BLE001
            if self.device.startswith("cuda"):
                self.device = "cpu"
                self.device_note = f"GPU inference failed ({exc.__class__.__name__}) - using CPU fallback."
                self.model.predictor = None
                self.model.predict(blank, **self._kwargs(0.25, 0.45, 320, None))
            else:
                raise ModelLoadError(f"Model warm-up failed: {exc}") from exc

    def _parse(self, result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(clss)
        return [
            Detection(self.names.get(int(c), str(c)), int(c), float(p), tuple(map(float, b)),
                      None if i is None else int(i))
            for b, p, c, i in zip(xyxy, confs, clss, ids)
        ]

    def _infer(self, frame, conf, iou, imgsz, classes, track=False, tracker="bytetrack.yaml") -> FrameResult:
        kw = self._kwargs(conf, iou, imgsz, classes)
        t0 = time.perf_counter()
        try:
            if track:
                results = self.model.track(frame, persist=True, tracker=tracker, **kw)
            else:
                results = self.model.predict(frame, **kw)
        except Exception as exc:  # noqa: BLE001
            if "out of memory" in str(exc).lower():
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001
                    pass
                raise InferenceError(
                    "GPU ran out of memory. Lower the image size, use YOLOv8n, or switch to CPU."
                ) from exc
            raise InferenceError(f"Inference failed: {exc}") from exc
        total_ms = (time.perf_counter() - t0) * 1000
        r = results[0]
        return FrameResult(self._parse(r), float(r.speed.get("inference", total_ms)), total_ms)

    # ------------------------------------------------------------------ public API
    def detect(self, frame, conf=0.25, iou=0.45, imgsz=640, classes=None) -> FrameResult:
        return self._infer(frame, conf, iou, imgsz, classes)

    def track(self, frame, conf=0.25, iou=0.45, imgsz=640, classes=None,
              tracker="bytetrack.yaml") -> FrameResult:
        return self._infer(frame, conf, iou, imgsz, classes, track=True, tracker=tracker)

    def reset_tracking(self) -> None:
        """Forget previous tracker state before starting a new video."""
        self.model.predictor = None


def annotate(frame: np.ndarray, detections: list[Detection], show_conf: bool = True) -> np.ndarray:
    """Draw bounding boxes, labels, confidence and (if present) tracking IDs."""
    out = frame.copy()
    h, w = out.shape[:2]
    thick = max(1, round(min(h, w) / 450))
    scale = max(0.4, min(h, w) / 1100)
    for d in detections:
        x1, y1, x2, y2 = map(int, d.box)
        color = color_for_class(d.class_id)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        text = (f"#{d.track_id} " if d.track_id is not None else "") + d.label
        if show_conf:
            text += f" {d.conf:.2f}"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(1, thick - 1))
        top = max(y1 - th - base - 2, 0)
        cv2.rectangle(out, (x1, top), (x1 + tw + 4, top + th + base + 2), color, -1)
        cv2.putText(out, text, (x1 + 2, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), max(1, thick - 1), cv2.LINE_AA)
    return out
