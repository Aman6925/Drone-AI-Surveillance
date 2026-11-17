"""Central configuration for the Aerial Vision prototype."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
ASSETS_DIR = BASE_DIR / "assets"
MODEL_DIR = BASE_DIR / "models"
for _d in (UPLOAD_DIR, OUTPUT_DIR, ASSETS_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

APP_TITLE = "Aerial Vision"
APP_SUBTITLE = "AI-Based Drone Surveillance & Object Detection Research Prototype"
DISCLAIMER = (
    "Research prototype built on a publicly available, COCO-pretrained YOLO model. "
    "It is not a production-grade or defence-grade surveillance system."
)

# Label shown in UI -> weights file name (auto-downloaded by Ultralytics on first use)
MODEL_OPTIONS = {
    "YOLOv8n - nano (fastest, recommended)": "yolov8n.pt",
    "YOLOv8s - small (more accurate, slower)": "yolov8s.pt",
}

# Published Ultralytics figures for COCO val2017 @ 640px. NOT measured by this app.
PUBLISHED_BENCHMARKS = {
    "yolov8n.pt": {"mAP50-95": 37.3, "params_M": 3.2, "GFLOPs": 8.7},
    "yolov8s.pt": {"mAP50-95": 44.9, "params_M": 11.2, "GFLOPs": 28.6},
}

IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
VIDEO_EXTENSIONS = ["mp4", "avi", "mov"]
MAX_IMAGE_MB = 25
MAX_VIDEO_MB = 500

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
IMGSZ_OPTIONS = [320, 480, 640, 800, 960, 1280]
DEFAULT_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]

TRACKERS = {"ByteTrack": "bytetrack.yaml", "BoT-SORT": "botsort.yaml"}
