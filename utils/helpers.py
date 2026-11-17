"""Small shared helpers and custom exceptions."""
from __future__ import annotations

from datetime import datetime

import cv2


class ModelLoadError(RuntimeError):
    """Model weights could not be loaded or downloaded."""


class InvalidMediaError(ValueError):
    """Uploaded image/video is unsupported, empty or corrupted."""


class InferenceError(RuntimeError):
    """Inference failed (e.g. GPU out of memory)."""


# BGR colour palette; colour is chosen per class id
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def color_for_class(class_id: int) -> tuple[int, int, int]:
    return PALETTE[int(class_id) % len(PALETTE)]


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def now_str() -> str:
    return datetime.now().strftime("%H:%M:%S")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def get_device_info() -> dict:
    """Describe the PyTorch / CUDA environment (never raises)."""
    info = {"torch": "not installed", "cuda_available": False, "cuda_version": "-",
            "gpu_name": "-", "vram_gb": 0.0}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = str(torch.version.cuda or "-")
        if info["cuda_available"]:
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["vram_gb"] = round(props.total_memory / 1024**3, 1)
    except Exception:  # noqa: BLE001 - informational only
        pass
    return info
