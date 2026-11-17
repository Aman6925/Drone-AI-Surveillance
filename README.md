# 🛰️ Aerial Vision - AI-Based Drone Surveillance & Object Detection

**YOLOv8 · OpenCV · PyTorch · Streamlit** - a research prototype that detects, counts and tracks objects in aerial images and videos.

> ⚠️ **Prototype notice:** built on publicly available, COCO-pretrained weights. It is **not** a production-grade or defence-grade surveillance system, and no accuracy figure in this project is claimed for real drone operations.

**GitHub description (copy/paste):**
> AI-based aerial surveillance prototype: YOLOv8 object detection, ByteTrack multi-object tracking and an analytics dashboard for images and drone video. Python, PyTorch, OpenCV, Streamlit. GPU inference with CPU fallback.

## 1. Introduction
Aerial Vision lets a user upload an image or video, runs a pretrained YOLOv8 model, draws labelled boxes with confidence scores, counts objects per category, optionally tracks them across frames, and shows everything in a dark Streamlit dashboard.

## 2. Problem statement
Drones capture far more footage than people can watch. Automating "what is in the frame, how many, and where did it go" reduces operator workload and enables searchable, quantified reports.

## 3. Objectives
- Detect people, cars, buses, trucks, motorcycles, bicycles (any of the 80 COCO classes).
- Image and frame-by-frame video processing with downloadable results.
- Persistent object IDs (ByteTrack / BoT-SORT).
- Analytics, honest evaluation notes, and GPU/CPU-aware optimisation.

## 4. System architecture
```
Upload (jpg/png/mp4/avi/mov) → validation (utils/file_handler.py)
   → OpenCV decode, one frame at a time → YOLOv8 inference (modules/detector.py, GPU FP16 or CPU)
   → confidence filter + NMS → [ByteTrack IDs + trails] (modules/tracker.py)
   → annotate boxes → OpenCV VideoWriter → H.264 re-encode (imageio-ffmpeg)
   → statistics (modules/analytics.py) → Streamlit dashboard (app.py)
```
```
Drone-AI-Surveillance/
├── app.py                  # Streamlit UI (6 pages)
├── config/settings.py      # paths, defaults, model list
├── modules/                # detector, image/video processors, tracker, analytics
├── utils/                  # file_handler (validation), helpers (device info, errors)
├── tests/                  # smoke + UI tests
├── uploads/  outputs/  models/  assets/
└── requirements.txt
```

## 5. Technologies
Python 3.11 · PyTorch · Ultralytics YOLOv8n/s · OpenCV · NumPy · Pandas · Matplotlib · Streamlit · imageio-ffmpeg · Git/GitHub

## 6. Installation (Windows 10/11, PowerShell)
```powershell
# 0) Check prerequisites
python --version        # 3.10 - 3.12 recommended (3.11 ideal)
nvidia-smi              # shows your RTX 2050 + driver CUDA version

# 1) Get the project
cd C:\Projects\Drone-AI-Surveillance

# 2) Virtual environment
python -m venv .venv
# if activation is blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 3) PyTorch WITH CUDA first (pip's default Windows wheel is CPU-only!)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4) Everything else
pip install -r requirements.txt

# 5) Verify GPU
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```
If step 5 prints `False`, the app still works on CPU (slower). Update your NVIDIA driver and re-run step 3.

## 7. Run
```powershell
streamlit run app.py          # opens http://localhost:8501
python tests/run_smoke_tests.py   # end-to-end tests (downloads yolov8n.pt on first run)
python tests/run_ui_test.py       # headless dashboard test
```
The first launch downloads `yolov8n.pt` (~6 MB) into `models/`. If you are offline, download it from the [Ultralytics assets releases](https://github.com/ultralytics/assets/releases) and drop it in `models/`.

## 8. How YOLO works
YOLO ("You Only Look Once") is a single-stage detector: one CNN forward pass predicts boxes, class scores and objectness for the whole image. A backbone extracts features, a neck (FPN/PAN) merges multi-scale features so small and large objects are both handled, and the head outputs boxes + classes (YOLOv8 is anchor-free). Overlapping duplicates are removed by **Non-Maximum Suppression (NMS)** using the IoU threshold; the **confidence threshold** drops weak boxes.

## 9. How OpenCV is used
`cv2.imdecode` (image decoding), `cv2.VideoCapture` (read frames one by one), `cv2.rectangle/putText/polylines` (boxes, labels, motion trails), `cv2.resize` (optional downscaling), `cv2.VideoWriter` (write result video), `cv2.imencode` (PNG download).

## 10. Object tracking
Detection answers "what is where **in this frame**". Tracking links detections **across frames** so each object keeps an ID. This project uses Ultralytics' **ByteTrack** (default) or **BoT-SORT**: a Kalman filter predicts each object's next position, and detections are matched to existing tracks by IoU (Hungarian assignment). ByteTrack also uses *low-confidence* detections in a second matching pass, which helps with partial occlusion. Tracking gives *unique object counts* instead of per-frame counts. Limitation: IDs can switch after long occlusions.

## 11. Model evaluation
- **Precision** = TP/(TP+FP), **Recall** = TP/(TP+FN), **mAP@50** (IoU≥0.5) and **mAP@50-95** (averaged over stricter IoUs).
- The app separates two kinds of numbers: **published** COCO benchmarks (YOLOv8n mAP50-95 = 37.3, YOLOv8s = 44.9 per Ultralytics, *not measured by us*) and **real measurements** (inference time/FPS on your machine; a coco8 sanity-check validation).
- No labelled aerial dataset is bundled, so **no aerial accuracy is claimed**.

## 12. Limitations
- COCO-pretrained → weak on very small, top-down objects; larger inference size helps but is slower.
- No fine-tuning on aerial data; no camera-motion compensation; tracking IDs may switch.
- Sandbox/CPU tests used synthetic panning video, not real drone footage.

## 13. Future improvements
Fine-tune on VisDrone / DOTA / UAVDT; SAHI sliced inference for tiny objects; export to ONNX/TensorRT; geo-referencing and zone-intrusion alerts; RTSP live streams; proper evaluation on a labelled aerial test set.

## 14. Screenshots
Add your own after running locally (put files in `assets/`):
`assets/dashboard.png` · `assets/image_detection.png` · `assets/tracking.png` · `assets/analytics.png`

## Public data for testing
Ultralytics' bundled `bus.jpg` (used by the demo button) · [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) (aerial images/videos, check licence) · [Pexels](https://www.pexels.com/search/videos/drone%20traffic/) / [Pixabay](https://pixabay.com/videos/search/drone%20traffic/) free drone footage.
