"""
Benchmark the RVM YOLO model on Raspberry Pi 5.

Measures: FPS, inference latency, CPU %, RAM usage.
Outputs:  Charts (PNG) + CSV + console summary.

Usage:
    python benchmark.py
"""

import sys
import os
import time
import csv

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import psutil
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np

from config import (
    get_model_path, CAM_INDEX, FRAME_W, FRAME_H,
    IMGSZ, CONF_THRESHOLD, IOU_THRESHOLD, MAX_DET,
    BENCHMARK_FRAMES, BENCHMARK_OUTPUT_DIR,
)
from ultralytics import YOLO


def run_benchmark():
    os.makedirs(BENCHMARK_OUTPUT_DIR, exist_ok=True)

    model_path = get_model_path()
    print(f"[Benchmark] Model: {model_path}")
    print(f"[Benchmark] Frames: {BENCHMARK_FRAMES}")
    print(f"[Benchmark] Resolution: {FRAME_W}x{FRAME_H} | imgsz={IMGSZ}")
    print()

    model = YOLO(model_path, task="detect")

    # Open camera
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2 if sys.platform == "linux" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("[Benchmark] ERROR: Cannot open camera")
        sys.exit(1)

    # Warm-up (5 frames)
    for _ in range(5):
        ret, frame = cap.read()
        if ret:
            model.predict(source=frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False)

    # Collect data
    latencies = []
    fps_list = []
    cpu_list = []
    ram_list = []

    process = psutil.Process(os.getpid())
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)

    print(f"[Benchmark] Running {BENCHMARK_FRAMES} frames ...")

    for i in range(BENCHMARK_FRAMES):
        ret, frame = cap.read()
        if not ret:
            print(f"  Frame {i}: camera read failed, skipping")
            continue

        t0 = time.perf_counter()
        model.predict(
            source=frame,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            max_det=MAX_DET,
            verbose=False,
        )
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0
        cpu = psutil.cpu_percent(interval=None)
        ram_mb = process.memory_info().rss / (1024 ** 2)

        latencies.append(latency_ms)
        fps_list.append(fps)
        cpu_list.append(cpu)
        ram_list.append(ram_mb)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{BENCHMARK_FRAMES}] latency={latency_ms:.1f}ms  fps={fps:.1f}  ram={ram_mb:.0f}MB")

    cap.release()

    # ---- Stats ----
    latencies = np.array(latencies)
    fps_arr = np.array(fps_list)
    ram_arr = np.array(ram_list)
    cpu_arr = np.array(cpu_list)

    summary = {
        "model": os.path.basename(model_path),
        "frames": len(latencies),
        "avg_fps": float(np.mean(fps_arr)),
        "min_fps": float(np.min(fps_arr)),
        "max_fps": float(np.max(fps_arr)),
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "avg_ram_mb": float(np.mean(ram_arr)),
        "peak_ram_mb": float(np.max(ram_arr)),
        "avg_cpu_pct": float(np.mean(cpu_arr)),
        "total_ram_gb": round(total_ram_gb, 1),
    }

    print()
    print("=" * 55)
    print(f"  RVM Benchmark — Raspberry Pi 5 ({summary['total_ram_gb']}GB)")
    print("=" * 55)
    print(f"  Model      : {summary['model']}")
    print(f"  Frames     : {summary['frames']}")
    print(f"  Avg FPS    : {summary['avg_fps']:.1f}")
    print(f"  Min FPS    : {summary['min_fps']:.1f}")
    print(f"  Max FPS    : {summary['max_fps']:.1f}")
    print(f"  Avg Latency: {summary['avg_latency_ms']:.1f} ms")
    print(f"  P95 Latency: {summary['p95_latency_ms']:.1f} ms")
    print(f"  Avg RAM    : {summary['avg_ram_mb']:.0f} MB")
    print(f"  Peak RAM   : {summary['peak_ram_mb']:.0f} MB")
    print(f"  Avg CPU    : {summary['avg_cpu_pct']:.1f}%")
    print("=" * 55)

    # ---- Save CSV ----
    csv_path = os.path.join(BENCHMARK_OUTPUT_DIR, "benchmark_data.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "latency_ms", "fps", "cpu_pct", "ram_mb"])
        for i in range(len(latencies)):
            writer.writerow([i + 1, f"{latencies[i]:.2f}", f"{fps_arr[i]:.2f}",
                             f"{cpu_arr[i]:.1f}", f"{ram_arr[i]:.1f}"])
    print(f"\n  CSV saved: {csv_path}")

    # ---- Charts ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"RVM Benchmark — RPi 5 ({summary['total_ram_gb']}GB RAM)", fontsize=14, fontweight="bold")
    fig.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#ccc")
        ax.xaxis.label.set_color("#ccc")
        ax.yaxis.label.set_color("#ccc")
        ax.title.set_color("#eee")
        for spine in ax.spines.values():
            spine.set_color("#333")

    x = np.arange(1, len(latencies) + 1)

    # FPS over time
    axes[0, 0].plot(x, fps_arr, color="#00b894", linewidth=1.2)
    axes[0, 0].axhline(y=np.mean(fps_arr), color="#fdcb6e", linestyle="--", linewidth=1, label=f"Avg: {np.mean(fps_arr):.1f}")
    axes[0, 0].set_title("FPS Over Time")
    axes[0, 0].set_xlabel("Frame")
    axes[0, 0].set_ylabel("FPS")
    axes[0, 0].legend(facecolor="#16213e", edgecolor="#333", labelcolor="#ccc")

    # Latency histogram
    axes[0, 1].hist(latencies, bins=25, color="#6c5ce7", edgecolor="#1a1a2e", alpha=0.85)
    axes[0, 1].axvline(x=np.mean(latencies), color="#fdcb6e", linestyle="--", linewidth=1, label=f"Avg: {np.mean(latencies):.1f}ms")
    axes[0, 1].set_title("Latency Distribution")
    axes[0, 1].set_xlabel("Latency (ms)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].legend(facecolor="#16213e", edgecolor="#333", labelcolor="#ccc")

    # RAM usage
    axes[1, 0].plot(x, ram_arr, color="#0984e3", linewidth=1.2)
    axes[1, 0].set_title("RAM Usage")
    axes[1, 0].set_xlabel("Frame")
    axes[1, 0].set_ylabel("RAM (MB)")

    # CPU usage
    axes[1, 1].plot(x, cpu_arr, color="#d63031", linewidth=1.2, alpha=0.8)
    axes[1, 1].set_title("CPU Usage")
    axes[1, 1].set_xlabel("Frame")
    axes[1, 1].set_ylabel("CPU %")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    chart_path = os.path.join(BENCHMARK_OUTPUT_DIR, "benchmark_charts.png")
    plt.savefig(chart_path, dpi=150, facecolor="#1a1a2e")
    plt.close()
    print(f"  Charts saved: {chart_path}")
    print("\n  Done!\n")


if __name__ == "__main__":
    run_benchmark()
