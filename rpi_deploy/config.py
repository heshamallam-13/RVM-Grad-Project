"""
Centralized configuration for RVM Detection on Raspberry Pi 5.
"""

import os

# =========================
# Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Model — prefer ONNX export, fall back to NCNN, then .pt
ONNX_MODEL_PATH = os.path.join(PROJECT_DIR, "rvm_best_yolov8s.onnx")
NCNN_MODEL_DIR = os.path.join(PROJECT_DIR, "rvm_best_yolov8s_ncnn_model")
PT_MODEL_PATH = os.path.join(PROJECT_DIR, "rvm_best_yolov8s.pt")
CLASS_NAMES_PATH = os.path.join(PROJECT_DIR, "class_names.txt")

def get_model_path():
    """Return ONNX model path if available."""
    if os.path.isfile(ONNX_MODEL_PATH):
        return ONNX_MODEL_PATH
    raise FileNotFoundError(
        f"ONNX model not found at {ONNX_MODEL_PATH}\n"
        "Run 'python export_model.py' on your PC first, then copy the .onnx file to the Pi."
    )

def get_class_names_path():
    return CLASS_NAMES_PATH

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
