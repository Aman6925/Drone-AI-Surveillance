"""Charts, benchmarking and (optional) validation helpers."""
from __future__ import annotations

import time
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import settings

BG, PANEL, FG, ACCENT = "#0B0F17", "#131A26", "#E6EDF3", "#00D1B2"
COLORS = ["#00D1B2", "#4C9AFF", "#FFB020", "#FF5C7A", "#A78BFA", "#7EE787", "#FF8A3D", "#5EEAD4"]


def _style(fig, ax):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=FG)
    for s in ax.spines.values():
        s.set_color("#2A3447")
    ax.title.set_color(FG)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def counts_df(counts: Counter) -> pd.DataFrame:
    df = pd.DataFrame(sorted(counts.items(), key=lambda kv: -kv[1]), columns=["object", "count"])
    return df


def bar_chart(counts: Counter, title: str = "Objects by category"):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    _style(fig, ax)
    if counts:
        df = counts_df(counts)
        bars = ax.bar(df["object"], df["count"], color=COLORS * 10)
        ax.bar_label(bars, color=FG, padding=2)
    ax.set_title(title)
    ax.set_ylabel("count")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def donut_chart(counts: Counter, title: str = "Class distribution"):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    fig.patch.set_facecolor(BG)
    if counts:
        df = counts_df(counts)
        ax.pie(df["count"], labels=df["object"], colors=COLORS * 10, startangle=90,
               wedgeprops=dict(width=0.42, edgecolor=BG), textprops=dict(color=FG, fontsize=8),
               autopct="%1.0f%%", pctdistance=0.78)
    ax.set_title(title, color=FG)
    fig.tight_layout()
    return fig


def conf_histogram(confs, bins: int = 20) -> np.ndarray:
    hist, _ = np.histogram(list(confs), bins=bins, range=(0, 1))
    return hist


def conf_chart(hist: np.ndarray, title: str = "Confidence distribution"):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    _style(fig, ax)
    n = len(hist)
    ax.bar((np.arange(n) + 0.5) / n, hist, width=0.9 / n, color=ACCENT)
    ax.set_xlim(0, 1)
    ax.set_xlabel("confidence")
    ax.set_ylabel("detections")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def timeline_chart(timeline: pd.DataFrame, title: str = "Objects per frame"):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    _style(fig, ax)
    ax.plot(timeline["frame"], timeline["objects"], color=ACCENT, linewidth=1.8)
    ax.fill_between(timeline["frame"], timeline["objects"], color=ACCENT, alpha=0.15)
    ax.set_xlabel("frame")
    ax.set_ylabel("objects")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def benchmark_inference(detector, image_bgr, runs: int = 10, conf=0.25, iou=0.45, imgsz=640) -> dict:
    """REAL measurement on this machine: warm-up then time `runs` inferences on one image."""
    for _ in range(2):
        detector.detect(image_bgr, conf, iou, imgsz)
    inf, tot = [], []
    for _ in range(runs):
        r = detector.detect(image_bgr, conf, iou, imgsz)
        inf.append(r.inference_ms)
        tot.append(r.total_ms)
    return {"runs": runs, "mean_inference_ms": float(np.mean(inf)), "min_inference_ms": float(np.min(inf)),
            "max_inference_ms": float(np.max(inf)), "mean_total_ms": float(np.mean(tot)),
            "fps": 1000.0 / float(np.mean(tot)), "imgsz": imgsz, "device": detector.device}


def run_validation(detector, data: str = "coco8.yaml", imgsz: int = 640) -> dict:
    """REAL measurement on a LABELLED dataset using Ultralytics' validator.
    coco8 has only 4 validation images, so treat the numbers as a sanity check, not a benchmark."""
    m = detector.model.val(data=data, imgsz=imgsz, device=detector.device, verbose=False, plots=False,
                           save=False, project=str(settings.OUTPUT_DIR / "val"), exist_ok=True)
    return {"precision": float(m.box.mp), "recall": float(m.box.mr), "mAP50": float(m.box.map50),
            "mAP50-95": float(m.box.map), "dataset": data, "imgsz": imgsz, "device": detector.device}


def history_df(history: list[dict]) -> pd.DataFrame:
    cols = ["time", "type", "source", "model", "device", "objects", "avg_conf", "inference_ms"]
    return pd.DataFrame(history, columns=cols) if history else pd.DataFrame(columns=cols)
