"""
Export the YOLOv8s model to ONNX format.
This script must be run on a PC where `ultralytics` can be installed.

Usage:
    python export_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import PT_MODEL_PATH, IMGSZ, PROJECT_DIR, CLASS_NAMES_PATH
from ultralytics import YOLO

def export():
    if not os.path.isfile(PT_MODEL_PATH):
        print(f"[Export] ERROR: Model file not found: {PT_MODEL_PATH}")
        sys.exit(1)

    print(f"[Export] Loading model: {PT_MODEL_PATH}")
    model = YOLO(PT_MODEL_PATH)

    print(f"[Export] Exporting to ONNX (imgsz={IMGSZ}) ...")
    model.export(format="onnx", imgsz=IMGSZ)

    print(f"[Export] Saving class names to: {CLASS_NAMES_PATH}")
    with open(CLASS_NAMES_PATH, "w") as f:
        for idx, name in model.names.items():
            f.write(f"{idx}:{name}\n")

    print("[Export] Done! ONNX model and class_names.txt saved.")
    print("[Export] Copy the generated '.onnx' file and 'class_names.txt' to the Pi.")

if __name__ == "__main__":
    export()
