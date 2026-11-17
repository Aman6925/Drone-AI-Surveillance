"""End-to-end smoke tests. Run from the project root:  python tests/run_smoke_tests.py"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from config import settings
from modules.detector import Detector
from modules.image_processor import run_image_detection
from modules.video_processor import get_video_info, process_video
from utils.helpers import InvalidMediaError, get_device_info
from utils import file_handler

results = []


def check(name, cond, extra=""):
    results.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")


class FakeUpload:  # mimics Streamlit's UploadedFile
    def __init__(self, name, data):
        self.name, self._d, self.size = name, data, len(data)

    def getvalue(self):
        return self._d

    def getbuffer(self):
        return self._d


def make_video(path, src, n=40):
    big = cv2.resize(src, None, fx=1.5, fy=1.5)
    h, w = src.shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
    for i in range(n):  # slow pan across the picture -> objects move between frames
        x = int(i * (big.shape[1] - w) / (n - 1))
        vw.write(big[0:h, x:x + w])
    vw.release()


def main():
    from ultralytics.utils import ASSETS
    img = cv2.imread(str(ASSETS / "bus.jpg"))
    print("Device info:", get_device_info())
    det = Detector("yolov8n.pt", "auto")
    print("Using device:", det.device, "-", det.device_note)
    check("model loads", det.model is not None)

    # --- image
    r = run_image_detection(det, img, 0.25, 0.45, 640, None)
    labels = set(r["counts"])
    check("image: detections found", len(r["detections"]) > 0, str(dict(r["counts"])))
    check("image: bus and person detected", {"bus", "person"} <= labels)
    check("image: annotated differs from original", not np.array_equal(r["annotated"], img))
    check("image: PNG encoding", len(file_handler.encode_image(r["annotated"])) > 1000)

    # --- empty result
    blank = np.zeros((480, 640, 3), np.uint8)
    r0 = run_image_detection(det, blank, 0.25, 0.45, 640, None)
    check("image: empty detections handled", len(r0["detections"]) == 0)

    # --- error handling
    for name, data in (("bad.jpg", b"not an image"), ("a.gif", b"GIF89a"), ("e.png", b"")):
        try:
            file_handler.load_image(FakeUpload(name, data))
            check(f"error handling: {name}", False)
        except InvalidMediaError as e:
            check(f"error handling: {name}", True, f"-> {e}")
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "broken.mp4"
        bad.write_bytes(b"\x00" * 2048)
        try:
            get_video_info(bad)
            check("error handling: corrupted video", False)
        except InvalidMediaError:
            check("error handling: corrupted video", True)

        # --- video
        vid = Path(tmp) / "pan.mp4"
        make_video(vid, img)
        check("video: info readable", get_video_info(vid)["frames"] >= 30)
        s = process_video(det, vid, 0.25, 0.45, 640, None, track=False, frame_stride=1)
        check("video: frames processed", s["frames_processed"] >= 30, f"({s['frames_processed']} frames, {s['proc_fps']:.1f} FPS)")
        check("video: output exists", s["path"].exists() and s["path"].stat().st_size > 1000, f"h264={s['h264']}")
        cap = cv2.VideoCapture(str(s["path"]))
        check("video: output re-openable", cap.isOpened() and cap.read()[0])
        cap.release()
        check("video: stats populated", s["total_detections"] > 0 and not s["timeline"].empty)

        # --- tracking
        t = process_video(det, vid, 0.25, 0.45, 640, None, track=True, tracker_cfg="bytetrack.yaml")
        check("tracking: IDs assigned", len(t["tracks_df"]) > 0, f"({len(t['tracks_df'])} tracks, {dict(t['unique_counts'])})")
        check("tracking: persistent IDs (some track spans >5 frames)", (t["tracks_df"]["frames_seen"] > 5).any())
        s["path"].unlink(missing_ok=True)
        t["path"].unlink(missing_ok=True)

    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
