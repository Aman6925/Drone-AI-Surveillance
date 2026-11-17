"""Aerial Vision - AI-based drone surveillance & object detection prototype (Streamlit dashboard)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from config import settings
from modules import analytics
from modules.detector import Detector
from modules.image_processor import detections_to_df, run_image_detection
from modules.video_processor import get_video_info, process_video
from utils import file_handler
from utils.helpers import (InferenceError, InvalidMediaError, ModelLoadError, bgr_to_rgb,
                           format_duration, get_device_info, now_str)

st.set_page_config(page_title=f"{settings.APP_TITLE} | Drone AI", page_icon="🛰️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1280px;}
.hero {background: linear-gradient(120deg,#0f2a3d 0%,#0b1f2e 45%,#10202a 100%);
       border:1px solid #1f3b4d; border-radius:14px; padding:22px 28px; margin-bottom:18px;}
.hero h1 {margin:0; font-size:2rem; letter-spacing:.5px;}
.hero p {margin:.3rem 0 0 0; color:#8fb4c9; font-size:1rem;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.75rem;
       background:#0e3b36; color:#5eead4; border:1px solid #14574f; margin-right:6px;}
div[data-testid="stMetric"] {background:#131A26; border:1px solid #223049; border-radius:10px; padding:12px 16px;}
.disclaimer {border-left:3px solid #FFB020; background:#1a1810; padding:8px 14px; border-radius:6px;
             color:#d9c28a; font-size:.85rem; margin-top:6px;}
</style>
""", unsafe_allow_html=True)

PAGES = ["📡 Dashboard", "🖼️ Image Detection", "🎞️ Video Detection", "🎯 Object Tracking",
         "📊 Analytics & Evaluation", "ℹ️ About Project"]


@dataclass
class Cfg:
    model_file: str
    device_pref: str
    conf: float
    iou: float
    imgsz: int
    classes: list[int] | None


# --------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading YOLO model (first run downloads the weights)...")
def load_detector(model_file: str, device_pref: str) -> Detector:
    return Detector(model_file, device_pref)


def show_fig(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><span class="pill">PROTOTYPE</span><span class="pill">YOLOv8</span>'
                f'<span class="pill">OpenCV</span><span class="pill">PyTorch</span>'
                f'<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def log_run(kind: str, source: str, det: Detector, n_objects: int, avg_conf: float, infer_ms: float) -> None:
    st.session_state.history.append({
        "time": now_str(), "type": kind, "source": source, "model": det.model_name,
        "device": det.device, "objects": n_objects, "avg_conf": round(avg_conf, 3),
        "inference_ms": round(infer_ms, 1)})


def sample_image_path():
    from ultralytics.utils import ASSETS
    return ASSETS / "bus.jpg"


def sidebar() -> tuple[str, Cfg]:
    st.sidebar.markdown("## 🛰️ AERIAL VISION")
    st.sidebar.caption("Drone AI Surveillance Prototype")
    page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.markdown("**Model settings**")
    label = st.sidebar.selectbox("Model", list(settings.MODEL_OPTIONS))
    device_pref = st.sidebar.selectbox("Inference device", ["auto", "cuda", "cpu"],
                                       help="auto = GPU if CUDA works, otherwise CPU fallback")
    conf = st.sidebar.slider("Confidence threshold", 0.05, 0.95, settings.DEFAULT_CONF, 0.05)
    iou = st.sidebar.slider("NMS IoU threshold", 0.10, 0.90, settings.DEFAULT_IOU, 0.05)
    imgsz = st.sidebar.select_slider("Inference size (px)", settings.IMGSZ_OPTIONS, settings.DEFAULT_IMGSZ,
                                     help="Larger sizes help with small aerial objects but are slower.")
    return page, Cfg(settings.MODEL_OPTIONS[label], device_pref, conf, iou, imgsz, None)


def class_filter(det: Detector, cfg: Cfg) -> None:
    names = det.names
    all_names = [names[i] for i in sorted(names)]
    use_all = st.sidebar.checkbox("Detect all 80 COCO classes", value=False)
    if use_all:
        cfg.classes = None
        return
    chosen = st.sidebar.multiselect("Object classes", all_names,
                                    default=[n for n in settings.DEFAULT_CLASSES if n in all_names])
    inv = {v: k for k, v in names.items()}
    cfg.classes = [inv[n] for n in chosen] or None


# --------------------------------------------------------------------------- pages
def page_dashboard(det: Detector, cfg: Cfg) -> None:
    hero(settings.APP_TITLE, settings.APP_SUBTITLE)
    dev = get_device_info()
    c = st.columns(4)
    c[0].metric("Model", det.model_name)
    c[1].metric("Inference device", "GPU (CUDA)" if det.device.startswith("cuda") else "CPU")
    c[2].metric("GPU", dev["gpu_name"] if dev["cuda_available"] else "None detected")
    c[3].metric("PyTorch", dev["torch"].split("+")[0])
    st.caption(det.device_note)

    hist = st.session_state.history
    df = analytics.history_df(hist)
    st.subheader("Session overview")
    m = st.columns(4)
    m[0].metric("Runs this session", len(df))
    m[1].metric("Images processed", int((df["type"] == "Image").sum()) if len(df) else 0)
    m[2].metric("Videos processed", int(df["type"].isin(["Video", "Tracking"]).sum()) if len(df) else 0)
    m[3].metric("Total detections", int(df["objects"].sum()) if len(df) else 0)

    left, right = st.columns([3, 2])
    with left:
        if len(df):
            st.dataframe(df.tail(8).iloc[::-1], hide_index=True)
        else:
            st.info("No runs yet. Open **Image Detection** or **Video Detection** to start, "
                    "or run the one-click demo below.")
    with right:
        totals = st.session_state.totals
        if totals:
            show_fig(analytics.donut_chart(totals, "All detections this session"))

    if st.button("▶ Run quick demo on sample image", type="primary"):
        try:
            img = cv2.imread(str(sample_image_path()))
            with st.spinner("Detecting..."):
                res = run_image_detection(det, img, cfg.conf, cfg.iou, cfg.imgsz, cfg.classes)
            res.update(name="bus.jpg (sample)", original=img)
            st.session_state.image_result = res
            record_image(det, res)
            st.success("Done - open **Image Detection** to see the annotated result.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Demo failed: {exc}")
    st.markdown(f'<div class="disclaimer">⚠️ {settings.DISCLAIMER}</div>', unsafe_allow_html=True)


def record_image(det: Detector, res: dict) -> None:
    n = len(res["detections"])
    avg = sum(res["confs"]) / n if n else 0.0
    log_run("Image", res["name"], det, n, avg, res["inference_ms"])
    st.session_state.totals.update(res["counts"])
    st.session_state.last_conf_hist = analytics.conf_histogram(res["confs"])


def page_image(det: Detector, cfg: Cfg) -> None:
    hero("Image Detection", "Detect and count objects in aerial or ground-level images.")
    up = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=settings.IMAGE_EXTENSIONS)
    use_sample = st.checkbox("Use the bundled sample image instead (Ultralytics bus.jpg)")
    if st.button("Run detection", type="primary", disabled=(up is None and not use_sample)):
        try:
            if up is not None:
                img, name = file_handler.load_image(up), up.name
            else:
                img, name = cv2.imread(str(sample_image_path())), "bus.jpg (sample)"
            with st.spinner("Running YOLO inference..."):
                res = run_image_detection(det, img, cfg.conf, cfg.iou, cfg.imgsz, cfg.classes)
            res.update(name=name, original=img)
            st.session_state.image_result = res
            record_image(det, res)
        except (InvalidMediaError, InferenceError) as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error: {exc}")
    res = st.session_state.get("image_result")
    if res:
        render_image_result(res)


def render_image_result(res: dict) -> None:
    dets = res["detections"]
    st.subheader(f"Result - {res['name']}")
    a, b = st.columns(2)
    a.image(bgr_to_rgb(res["original"]), caption="Original")
    b.image(bgr_to_rgb(res["annotated"]), caption="Detections")
    n = len(dets)
    m = st.columns(4)
    m[0].metric("Total objects", n)
    m[1].metric("Avg confidence", f"{sum(res['confs']) / n:.2f}" if n else "-")
    m[2].metric("Model inference", f"{res['inference_ms']:.1f} ms")
    m[3].metric("Total pipeline", f"{res['total_ms']:.1f} ms")
    if n == 0:
        st.info("No objects found. Try lowering the confidence threshold, enabling more classes, "
                "or increasing the inference size (small aerial objects need larger sizes).")
    else:
        c1, c2 = st.columns(2)
        with c1:
            show_fig(analytics.bar_chart(res["counts"]))
        with c2:
            st.markdown("**Category-wise counts**")
            st.dataframe(analytics.counts_df(res["counts"]), hide_index=True)
        with st.expander("All detections (label, confidence, box)"):
            st.dataframe(detections_to_df(dets), hide_index=True)
    st.download_button("⬇ Download processed image (PNG)", file_handler.encode_image(res["annotated"]),
                       file_name="detections.png", mime="image/png")


def page_video(det: Detector, cfg: Cfg, tracking: bool) -> None:
    mode = "Tracking" if tracking else "Video"
    hero("Object Tracking" if tracking else "Video Detection",
         "Assign persistent IDs to objects across frames (ByteTrack / BoT-SORT)." if tracking
         else "Frame-by-frame detection on uploaded video, streamed to keep RAM low.")
    up = st.file_uploader("Upload a video (MP4, AVI, MOV)", type=settings.VIDEO_EXTENSIONS,
                          key=f"up_{mode}")
    path = None
    if up is not None:
        key = (up.name, up.size)
        if st.session_state.get(f"vkey_{mode}") != key:
            try:
                st.session_state[f"vpath_{mode}"] = file_handler.save_video_upload(up)
                st.session_state[f"vkey_{mode}"] = key
                st.session_state[f"vinfo_{mode}"] = get_video_info(st.session_state[f"vpath_{mode}"])
            except InvalidMediaError as exc:
                st.session_state.pop(f"vkey_{mode}", None)
                st.error(str(exc))
        if st.session_state.get(f"vkey_{mode}") == key:
            path = st.session_state[f"vpath_{mode}"]
            i = st.session_state[f"vinfo_{mode}"]
            st.caption(f"{i['width']}x{i['height']} px · {i['fps']:.1f} FPS · {i['frames']} frames · "
                       f"{format_duration(i['duration_s'])}")

    o = st.columns(4)
    stride = o[0].number_input("Process every Nth frame", 1, 10, 1)
    max_frames = o[1].number_input("Max frames (0 = all)", 0, 100000, 300, step=50)
    max_side = o[2].selectbox("Downscale longest side", [0, 640, 960, 1280], index=2,
                              format_func=lambda v: "original" if v == 0 else f"{v}px")
    tracker_name, trails = "ByteTrack", True
    if tracking:
        tracker_name = o[3].selectbox("Tracker", list(settings.TRACKERS))
        trails = st.checkbox("Draw motion trails", value=True)

    if st.button("Process video", type="primary", disabled=path is None):
        bar, preview = st.progress(0.0, text="Starting..."), st.empty()

        def on_progress(done, total):
            frac = min(done / total, 1.0) if total else 0.0
            bar.progress(frac, text=f"Processed {done}" + (f" / {total}" if total else "") + " frames")

        try:
            stats = process_video(
                det, path, cfg.conf, cfg.iou, cfg.imgsz, cfg.classes, track=tracking,
                tracker_cfg=settings.TRACKERS[tracker_name], frame_stride=int(stride),
                max_frames=int(max_frames), max_side=int(max_side), trails=trails,
                progress_cb=on_progress,
                preview_cb=lambda f, idx: preview.image(bgr_to_rgb(f), caption=f"Live preview - frame {idx}"))
            preview.empty()
            bar.progress(1.0, text="Finished")
            st.session_state[f"vres_{mode}"] = stats
            n = stats["total_detections"]
            log_run(mode, up.name if up else "video", det, n, stats["avg_conf"], stats["avg_inference_ms"])
            st.session_state.totals.update(stats["unique_counts"] if tracking else stats["counts"])
            st.session_state.last_conf_hist = stats["conf_hist"]
            st.session_state.last_timeline = stats["timeline"]
        except (InvalidMediaError, InferenceError) as exc:
            bar.empty()
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            st.error(f"Unexpected error while processing the video: {exc}")

    stats = st.session_state.get(f"vres_{mode}")
    if stats:
        render_video_result(stats, tracking)


def render_video_result(s: dict, tracking: bool) -> None:
    st.subheader("Result")
    m = st.columns(5)
    m[0].metric("Frames processed", s["frames_processed"])
    m[1].metric("Processing speed", f"{s['proc_fps']:.1f} FPS")
    m[2].metric("Avg inference", f"{s['avg_inference_ms']:.1f} ms")
    m[3].metric("Unique objects" if tracking else "Total detections",
                sum(s["unique_counts"].values()) if tracking else s["total_detections"])
    m[4].metric("Elapsed", format_duration(s["elapsed_s"]))
    st.video(str(s["path"]))
    if not s["h264"]:
        st.warning("H.264 re-encoding was unavailable, so in-browser playback may fail. "
                   "The downloaded file will still play in VLC / Windows Media Player.")
    if s["total_detections"] == 0:
        st.info("No objects were detected in the processed frames. Try lowering the confidence threshold.")
    else:
        counts = s["unique_counts"] if tracking else s["counts"]
        c1, c2 = st.columns(2)
        with c1:
            show_fig(analytics.bar_chart(counts, "Unique objects (by track ID)" if tracking
                                         else "Detections by category (summed over frames)"))
        with c2:
            show_fig(analytics.timeline_chart(s["timeline"]))
        if not tracking:
            st.caption("Without tracking, one object visible for 100 frames counts as 100 detections. "
                       "Use the **Object Tracking** page to count unique objects.")
    if tracking and len(s["tracks_df"]):
        st.markdown("**Track table** (one row per tracking ID)")
        st.dataframe(s["tracks_df"], hide_index=True)
        st.download_button("⬇ Download tracks (CSV)", s["tracks_df"].to_csv(index=False),
                           file_name="tracks.csv", mime="text/csv")
    with open(s["path"], "rb") as f:
        st.download_button("⬇ Download processed video (MP4)", f.read(), file_name=s["path"].name,
                           mime="video/mp4")


def page_analytics(det: Detector, cfg: Cfg) -> None:
    hero("Analytics & Evaluation", "Session statistics, honest model evaluation and optimisation notes.")
    t1, t2, t3 = st.tabs(["Session analytics", "Model evaluation", "Optimization"])

    with t1:
        df = analytics.history_df(st.session_state.history)
        if df.empty:
            st.info("Run an image or video first - analytics are computed from your actual runs.")
        else:
            totals = st.session_state.totals
            m = st.columns(4)
            m[0].metric("Total detected objects", sum(totals.values()))
            m[1].metric("Distinct categories", len(totals))
            m[2].metric("Mean confidence", f"{df['avg_conf'].replace(0, float('nan')).mean():.2f}")
            m[3].metric("Mean inference", f"{df['inference_ms'].mean():.1f} ms")
            c1, c2 = st.columns(2)
            with c1:
                show_fig(analytics.bar_chart(totals, "Category-wise object count"))
            with c2:
                show_fig(analytics.donut_chart(totals, "Object distribution"))
            c3, c4 = st.columns(2)
            with c3:
                if st.session_state.get("last_conf_hist") is not None:
                    show_fig(analytics.conf_chart(st.session_state.last_conf_hist, "Confidence (latest run)"))
            with c4:
                if st.session_state.get("last_timeline") is not None:
                    show_fig(analytics.timeline_chart(st.session_state.last_timeline, "Objects per frame (latest video)"))
            st.markdown("**Run history**")
            st.dataframe(df, hide_index=True)
            st.download_button("⬇ Download history (CSV)", df.to_csv(index=False), file_name="history.csv")

    with t2:
        st.markdown("""
**Metrics explained**
- **Precision** = TP / (TP + FP): of everything the model *predicted*, how much was correct.
- **Recall** = TP / (TP + FN): of everything that was *really there*, how much was found.
- **mAP** (mean Average Precision): area under the precision-recall curve, averaged over classes.
  *mAP@50* uses IoU ≥ 0.5 for a match; *mAP@50-95* averages IoU thresholds 0.50 to 0.95 (stricter).
- **Confidence threshold**: minimum score to keep a box. Higher = fewer false positives, more misses.
- **Inference time**: milliseconds for one forward pass; FPS ≈ 1000 / total ms.
""")
        st.markdown("##### 1) Published benchmark (NOT measured by this app)")
        bench = pd.DataFrame(settings.PUBLISHED_BENCHMARKS).T.rename(
            columns={"params_M": "Params (M)", "GFLOPs": "GFLOPs"})
        bench.insert(0, "Dataset", "COCO val2017 @640px")
        st.dataframe(bench)
        st.caption("Figures published by Ultralytics for the pretrained COCO models. "
                   "They say nothing about drone/aerial footage, where accuracy is typically lower.")
        st.markdown("##### 2) Your measurements (real, on this machine)")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("⏱ Benchmark inference speed (10 runs)"):
                with st.spinner("Benchmarking..."):
                    img = cv2.imread(str(sample_image_path()))
                    st.session_state.bench = analytics.benchmark_inference(det, img, 10, cfg.conf, cfg.iou, cfg.imgsz)
            b = st.session_state.get("bench")
            if b:
                st.success(f"Mean inference **{b['mean_inference_ms']:.1f} ms** "
                           f"(min {b['min_inference_ms']:.1f} / max {b['max_inference_ms']:.1f}), "
                           f"end-to-end **{b['fps']:.1f} FPS** @ {b['imgsz']}px on `{b['device']}`.")
        with c2:
            st.caption("Validation needs a *labelled* dataset. `coco8` (4 val images, auto-downloaded) "
                       "is only a sanity check - not a meaningful accuracy estimate.")
            if st.button("🧪 Validate on coco8 (sanity check)"):
                try:
                    with st.spinner("Running validation (downloads coco8 on first use)..."):
                        st.session_state.val = analytics.run_validation(det, "coco8.yaml", cfg.imgsz)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Validation failed (internet needed for the dataset): {exc}")
            v = st.session_state.get("val")
            if v:
                st.dataframe(pd.DataFrame([{"Precision": v["precision"], "Recall": v["recall"],
                                            "mAP50": v["mAP50"], "mAP50-95": v["mAP50-95"]}]).round(3),
                             hide_index=True)
                st.caption(f"Measured on {v['dataset']} (tiny, partly seen in training) - do not quote as model accuracy.")

    with t3:
        dev = get_device_info()
        st.markdown(f"""
**Current setup:** `{det.model_name}` on `{det.device}` · PyTorch `{dev['torch']}` · CUDA runtime `{dev['cuda_version']}`
· GPU `{dev['gpu_name']}` ({dev['vram_gb']} GB)

**Optimisations used in this project**
- Lightweight model (YOLOv8n, ~3.2M parameters) that fits a 4 GB GPU comfortably.
- Automatic GPU inference with **FP16 (half precision)** on CUDA, with automatic CPU fallback.
- Streaming video pipeline (one frame in RAM at a time), optional frame skipping and downscaling.
- Adjustable confidence / IoU thresholds and inference size.

**Future deployment options (not implemented here)**
```python
from ultralytics import YOLO
model = YOLO("models/yolov8n.pt")
model.export(format="onnx")                 # portable, runs with ONNX Runtime (CPU/GPU/DirectML)
model.export(format="engine", half=True)    # TensorRT engine - needs NVIDIA TensorRT installed
```
ONNX makes the model framework-independent; TensorRT fuses layers and optimises kernels for a specific NVIDIA GPU,
usually giving lower latency. Speed-ups depend on hardware, so measure before claiming any number.
""")


def page_about() -> None:
    hero("About this project", settings.APP_SUBTITLE)
    st.markdown(f'<div class="disclaimer">⚠️ {settings.DISCLAIMER}</div>', unsafe_allow_html=True)
    st.markdown("""
### Problem
Drones produce huge volumes of aerial footage that humans cannot watch continuously. Automatic detection
and tracking can flag people and vehicles, count them and highlight activity for an operator.

### Pipeline
""")
    st.graphviz_chart("""digraph G { rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#131A26", color="#00D1B2", fontcolor="#E6EDF3"];
      edge [color="#8fb4c9"];
      Upload -> "OpenCV decode\\n(frame by frame)" -> "YOLOv8 (PyTorch)\\nGPU / CPU" -> "NMS + threshold" ->
      "ByteTrack IDs\\n(optional)" -> "Annotate + encode" -> "Streamlit dashboard\\n+ analytics"; }""")
    st.markdown("""
### Stack
Python · PyTorch · Ultralytics YOLOv8 · OpenCV · NumPy · Pandas · Matplotlib · Streamlit

### Known limitations
- COCO-pretrained model: trained mostly on ground-level photos, so **small objects seen from high altitude are often missed**.
  Fine-tuning on an aerial dataset such as VisDrone would be the next step.
- Tracking can swap IDs during occlusion or fast camera motion; no camera-motion compensation beyond what BoT-SORT offers.
- Evaluation figures shown are either published COCO benchmarks or tiny sanity checks, clearly labelled as such.
""")


# --------------------------------------------------------------------------- main
def main() -> None:
    for key, default in (("history", []), ("totals", Counter())):
        if key not in st.session_state:
            st.session_state[key] = default
    if "cleaned" not in st.session_state:
        file_handler.cleanup_old_files(settings.UPLOAD_DIR)
        file_handler.cleanup_old_files(settings.OUTPUT_DIR)
        st.session_state.cleaned = True

    page, cfg = sidebar()
    if page == PAGES[5]:
        page_about()
        return
    try:
        det = load_detector(cfg.model_file, cfg.device_pref)
    except ModelLoadError as exc:
        st.error(f"Model could not be loaded.\n\n{exc}")
        st.stop()
    class_filter(det, cfg)
    st.sidebar.divider()
    st.sidebar.caption(f"Device: **{det.device}**")

    if page == PAGES[0]:
        page_dashboard(det, cfg)
    elif page == PAGES[1]:
        page_image(det, cfg)
    elif page == PAGES[2]:
        page_video(det, cfg, tracking=False)
    elif page == PAGES[3]:
        page_video(det, cfg, tracking=True)
    else:
        page_analytics(det, cfg)


main()
