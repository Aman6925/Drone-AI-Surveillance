"""Upload validation, saving and image decoding."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from config import settings
from utils.helpers import InvalidMediaError


def _extension(name: str) -> str:
    return Path(name).suffix.lower().lstrip(".")


def validate_upload(uploaded_file, allowed: list[str], max_mb: int) -> None:
    ext = _extension(uploaded_file.name)
    if ext not in allowed:
        raise InvalidMediaError(
            f"Unsupported file type '.{ext}'. Allowed: {', '.join('.' + a for a in allowed)}."
        )
    size_mb = uploaded_file.size / 1024**2
    if uploaded_file.size == 0:
        raise InvalidMediaError("The uploaded file is empty.")
    if size_mb > max_mb:
        raise InvalidMediaError(f"File is {size_mb:.1f} MB; the limit is {max_mb} MB.")


def load_image(uploaded_file) -> np.ndarray:
    """Decode an uploaded image to a BGR array, raising InvalidMediaError if corrupt."""
    validate_upload(uploaded_file, settings.IMAGE_EXTENSIONS, settings.MAX_IMAGE_MB)
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise InvalidMediaError("The image could not be decoded - it may be corrupted.")
    return img


def save_video_upload(uploaded_file) -> Path:
    validate_upload(uploaded_file, settings.VIDEO_EXTENSIONS, settings.MAX_VIDEO_MB)
    path = settings.UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{Path(uploaded_file.name).name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return path


def encode_image(img_bgr: np.ndarray, ext: str = ".png") -> bytes:
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise InvalidMediaError("Could not encode the processed image.")
    return buf.tobytes()


def cleanup_old_files(directory: Path, max_age_hours: float = 6.0) -> None:
    """Delete stale temporary files so uploads/outputs do not fill the disk."""
    cutoff = time.time() - max_age_hours * 3600
    for p in Path(directory).glob("*"):
        try:
            if p.is_file() and p.name != ".gitkeep" and p.stat().st_mtime < cutoff:
                p.unlink()
        except OSError:
            pass
