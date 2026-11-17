"""Streaming video pipeline: reads one frame at a time so RAM use stays flat for long videos."""
from __future__ import annotations

import subprocess
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from config import settings
from modules.detector import Detector, annotate
from modules.tracker import TrackStats, TrailDrawer
from utils.helpers import InvalidMediaError, timestamp

HIST_BINS = 20


def get_video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise InvalidMediaError("The video could not be opened - it may be corrupted or use an unsupported codec.")
    ok, _ = cap.read()
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    if not ok or info["width"] == 0:
        raise InvalidMediaError("No readable frames were found in this video.")
    info["duration_s"] = info["frames"] / info["fps"] if info["frames"] > 0 else 0.0
    return info


def _resize(frame: np.ndarray, max_side: int) -> np.ndarray:
    if not max_side:
        return frame
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    return frame if s >= 1 else cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def _to_h264(raw: Path, final: Path) -> bool:
    """Re-encode to H.264 so browsers can play the result. Returns False if ffmpeg is unavailable."""
    try:
        import imageio_ffmpeg

        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(raw), "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-movflags", "+faststart", "-an", str(final)]
        subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
        raw.unlink(missing_ok=True)
        return True
    except Exception:  # noqa: BLE001
        raw.replace(final)
        return False


def process_video(detector: Detector, in_path: Path, conf: float, iou: float, imgsz: int,
                  classes: list[int] | None, track: bool = False, tracker_cfg: str = "bytetrack.yaml",
                  frame_stride: int = 1, max_frames: int = 0, max_side: int = 0, trails: bool = True,
                  progress_cb=None, preview_cb=None, preview_every: int = 5) -> dict:
    info = get_video_info(in_path)
    cap = cv2.VideoCapture(str(in_path))
    stem = Path(in_path).stem
    raw = settings.OUTPUT_DIR / f"_raw_{stem}.mp4"
    final = settings.OUTPUT_DIR / f"processed_{stem}_{timestamp()}.mp4"
    out_fps = max(1.0, info["fps"] / max(1, frame_stride))

    expected = None
    if info["frames"] > 0:
        expected = -(-info["frames"] // max(1, frame_stride))
        if max_frames:
            expected = min(expected, max_frames)

    if track:
        detector.reset_tracking()
    stats_tracks, drawer = TrackStats(), TrailDrawer()
    counts: Counter = Counter()
    hist = np.zeros(HIST_BINS, dtype=np.int64)
    conf_sum, conf_n, infer_sum, total_sum = 0.0, 0, 0.0, 0.0
    rows: list[dict] = []
    writer = None
    read_idx, done = 0, 0
    t_start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            read_idx += 1
            if (read_idx - 1) % max(1, frame_stride):
                continue
            if max_frames and done >= max_frames:
                break
            frame = _resize(frame, max_side)
            res = (detector.track(frame, conf, iou, imgsz, classes, tracker_cfg) if track
                   else detector.detect(frame, conf, iou, imgsz, classes))
            out = annotate(frame, res.detections)
            if track:
                stats_tracks.update(res.detections, read_idx)
                if trails:
                    drawer.update(res.detections)
                    drawer.draw(out)
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
                if not writer.isOpened():
                    raise InvalidMediaError("Could not create the output video file.")
            writer.write(out)

            done += 1
            infer_sum += res.inference_ms
            total_sum += res.total_ms
            row = {"frame": read_idx, "time_s": round(read_idx / info["fps"], 2),
                   "objects": len(res.detections)}
            for d in res.detections:
                counts[d.label] += 1
                row[d.label] = row.get(d.label, 0) + 1
                conf_sum += d.conf
                conf_n += 1
                hist[min(int(d.conf * HIST_BINS), HIST_BINS - 1)] += 1
            rows.append(row)
            if progress_cb:
                progress_cb(done, expected)
            if preview_cb and done % preview_every == 0:
                preview_cb(out, read_idx)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if done == 0:
        raw.unlink(missing_ok=True)
        raise InvalidMediaError("No frames could be processed from this video.")

    elapsed = time.perf_counter() - t_start
    h264 = _to_h264(raw, final)
    timeline = pd.DataFrame(rows).fillna(0)
    for c in timeline.columns:
        if c != "time_s":
            timeline[c] = timeline[c].astype(int)
    return {
        "path": final, "h264": h264, "info": info, "tracking": track,
        "frames_processed": done, "elapsed_s": elapsed, "proc_fps": done / elapsed if elapsed else 0.0,
        "avg_inference_ms": infer_sum / done, "avg_total_ms": total_sum / done,
        "total_detections": int(sum(counts.values())), "counts": counts,
        "avg_conf": conf_sum / conf_n if conf_n else 0.0, "conf_hist": hist,
        "timeline": timeline,
        "tracks_df": stats_tracks.to_df() if track else pd.DataFrame(),
        "unique_counts": stats_tracks.unique_counts() if track else Counter(),
    }
