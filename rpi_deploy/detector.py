"""
Optimized detection engine for Raspberry Pi 5.
Uses OpenCV DNN with an ONNX YOLOv8 model for best ARM performance
without needing the heavy PyTorch/Ultralytics dependencies.
"""

import time
import os
import cv2
import numpy as np
import threading

from config import (
    get_model_path, get_class_names_path, CAM_INDEX, FRAME_W, FRAME_H, TARGET_FPS,
    IMGSZ, CONF_THRESHOLD, IOU_THRESHOLD, MAX_DET, INFER_EVERY_N,
    PET_CLASSES, CAN_CLASSES, normalize_name,
)

class Detector:
    """Manages camera capture and OpenCV DNN YOLO inference."""

    def __init__(self, model_path=None):
        path = model_path or get_model_path()
        print(f"[Detector] Loading ONNX model: {path}")
        
        # Initialize OpenCV DNN Network
        self.net = cv2.dnn.readNetFromONNX(path)
        print("[Detector] Model loaded.")

        # Try to load class names
        self.class_names = {}
        class_path = get_class_names_path()
        if os.path.exists(class_path):
            with open(class_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        idx_str, name = line.split(":", 1)
                        self.class_names[int(idx_str)] = name
            print(f"[Detector] Loaded {len(self.class_names)} class names.")
        else:
            print(f"[Detector] WARNING: Class names file not found at {class_path}.")
            print("Run export_model.py on PC first, then copy class_names.txt to the Pi.")

        self.cap = None
        self.frame_count = 0
        self.last_annotated = None
        
        # Threaded Camera
        self.current_frame = None
        self.thread_running = False
        self.camera_thread = None

        # FPS smoothing
        self._prev_time = 0.0
        self._fps_smooth = 0.0
        self._alpha = 0.15

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    def open_camera(self):
        """Open the webcam. Returns True on success."""
        if self.cap and self.cap.isOpened():
            return True

        # Try V4L2 first (Linux/RPi), then default backend
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            self.cap = cv2.VideoCapture(CAM_INDEX, backend)
            if self.cap.isOpened():
                break

        if not self.cap.isOpened():
            print("[Detector] ERROR: Cannot open camera")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._prev_time = time.time()
        print(f"[Detector] Camera opened (index={CAM_INDEX})")
        
        # Start background reading thread
        self.thread_running = True
        self.camera_thread = threading.Thread(target=self._update_frame_loop, daemon=True)
        self.camera_thread.start()
        
        return True

    def _update_frame_loop(self):
        """Continuously reads frames from the camera in the background."""
        while self.thread_running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame
            # Small sleep to prevent 100% CPU usage
            time.sleep(0.01)

    def release_camera(self):
        """Release the webcam."""
        self.thread_running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            print("[Detector] Camera released")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def read_and_detect(self):
        """
        Read one frame and optionally run inference.

        Returns dict:
            {
                "ok": bool,
                "frame_jpeg": bytes | None,
                "detected_type": "pet" | "can" | "none",
                "detected_conf": float,
                "fps": float,
            }
        """
        if not self.thread_running or self.current_frame is None:
            return {"ok": False, "frame_jpeg": None, "detected_type": "none",
                    "detected_conf": 0.0, "fps": 0.0}

        # Grab a copy of the latest background frame
        frame = self.current_frame.copy()

        self.frame_count += 1
        detected_type = "none"
        detected_conf = 0.0

        do_infer = (self.frame_count % INFER_EVERY_N == 0)

        if do_infer:
            # 1. OpenCV DNN Preprocessing (BGR to RGB, scale to 0-1, resize to IMGSZ)
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (IMGSZ, IMGSZ), swapRB=True, crop=False)
            self.net.setInput(blob)

            # 2. Inference
            outputs = self.net.forward()

            # 3. Post-processing
            # YOLOv8 ONNX output shape: (1, num_classes + 4, num_boxes)
            preds = outputs[0]

            # Transpose to (num_boxes, num_classes + 4) for easier parsing
            if preds.shape[0] < preds.shape[1]:
                preds = preds.T

            boxes = []
            scores = []
            class_ids = []

            # Calculate scaling factors to map back to original frame size
            x_factor = frame.shape[1] / IMGSZ
            y_factor = frame.shape[0] / IMGSZ

            # Parse predictions
            for row in preds:
                cls_scores = row[4:]
                class_id = np.argmax(cls_scores)
                score = cls_scores[class_id]

                if score > CONF_THRESHOLD:
                    cx, cy, w, h = row[0], row[1], row[2], row[3]

                    # Convert to top-left x,y,w,h for OpenCV
                    left = int((cx - w / 2) * x_factor)
                    top = int((cy - h / 2) * y_factor)
                    width = int(w * x_factor)
                    height = int(h * y_factor)

                    boxes.append([left, top, width, height])
                    scores.append(float(score))
                    class_ids.append(class_id)

            # Apply Non-Maximum Suppression
            indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)

            annotated = frame.copy()

            if len(indices) > 0:
                # Handle difference in OpenCV return format for NMSBoxes
                if isinstance(indices, tuple):
                    indices = indices[0]
                else:
                    indices = indices.flatten()

                # Limit max detections
                indices = indices[:MAX_DET]

                for i in indices:
                    box = boxes[i]
                    left, top, width, height = box
                    conf = scores[i]
                    cls_id = class_ids[i]

                    cls_name = self.class_names.get(cls_id, str(cls_id))
                    norm_name = normalize_name(cls_name)

                    if norm_name in PET_CLASSES and conf > detected_conf:
                        detected_type = "pet"
                        detected_conf = conf
                    elif norm_name in CAN_CLASSES and conf > detected_conf:
                        detected_type = "can"
                        detected_conf = conf

                    # Draw bounding box and label
                    cv2.rectangle(annotated, (left, top), (left + width, top + height), (0, 255, 0), 2)
                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(annotated, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self.last_annotated = annotated
        else:
            annotated = self.last_annotated if self.last_annotated is not None else frame

        # FPS calculation
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now
        instant_fps = 1.0 / dt if dt > 0 else 0.0
        self._fps_smooth = (1 - self._alpha) * self._fps_smooth + self._alpha * instant_fps

        # Overlay FPS on frame
        cv2.putText(
            annotated,
            f"FPS: {self._fps_smooth:.1f}",
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
        )

        # Encode to JPEG for streaming
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])

        return {
            "ok": True,
            "frame_jpeg": jpeg.tobytes(),
            "detected_type": detected_type,
            "detected_conf": detected_conf,
            "fps": round(self._fps_smooth, 1),
        }

    @property
    def fps(self):
        return round(self._fps_smooth, 1)
