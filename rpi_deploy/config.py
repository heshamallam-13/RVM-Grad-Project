"""
Centralized configuration for RVM Detection on Raspberry Pi 5.
"""

import os

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Model — prefer NCNN export, fall back to .pt
NCNN_MODEL_DIR = os.path.join(PROJECT_DIR, "rvm_best_yolov8s_ncnn_model")
PT_MODEL_PATH = os.path.join(PROJECT_DIR, "rvm_best_yolov8s.pt")

def get_model_path():
    """Return NCNN model path if available, otherwise .pt path."""
    if os.path.isdir(NCNN_MODEL_DIR):
        return NCNN_MODEL_DIR
    return PT_MODEL_PATH

# =========================
# Camera
# =========================
CAM_INDEX = 0          # 0 = USB webcam on RPi
FRAME_W = 640
FRAME_H = 480
TARGET_FPS = 30

# =========================
# Detection
# =========================
IMGSZ = 640
CONF_THRESHOLD = 0.80
IOU_THRESHOLD = 0.50
MAX_DET = 50
INFER_EVERY_N = 1      # run inference on every Nth frame (increase to skip frames)

# =========================
# Reward Points
# =========================
PET_POINTS = 50
CAN_POINTS = 100

# =========================
# Class Name Mapping
# =========================
PET_CLASSES = {"pet", "pet_bottle", "plastic_bottle", "bottle", "plastic"}
CAN_CLASSES = {"can", "aluminum_can", "aluminium_can", "tin_can", "aluminum"}

def normalize_name(name: str) -> str:
    """Normalize class name for matching."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")

# =========================
# Server
# =========================
HOST = "0.0.0.0"
PORT = 5000

# =========================
# Benchmark
# =========================
BENCHMARK_FRAMES = 100
BENCHMARK_OUTPUT_DIR = os.path.join(BASE_DIR, "benchmark_results")
