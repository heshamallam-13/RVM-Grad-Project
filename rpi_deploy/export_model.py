"""
Export the YOLOv8s model to NCNN format for Raspberry Pi ARM CPU.

Usage:
    python export_model.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import PT_MODEL_PATH, IMGSZ
from ultralytics import YOLO


def export():
    if not os.path.isfile(PT_MODEL_PATH):
        print(f"[Export] ERROR: Model file not found: {PT_MODEL_PATH}")
        sys.exit(1)

    print(f"[Export] Loading model: {PT_MODEL_PATH}")
    model = YOLO(PT_MODEL_PATH)

    print(f"[Export] Exporting to NCNN (imgsz={IMGSZ}) ...")
    model.export(format="ncnn", imgsz=IMGSZ)

    print("[Export] Done! NCNN model saved alongside the .pt file.")
    print("[Export] The detector will auto-detect and use the NCNN model.")


if __name__ == "__main__":
    export()
