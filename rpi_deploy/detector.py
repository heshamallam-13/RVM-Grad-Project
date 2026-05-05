"""
Optimized detection engine for Raspberry Pi 5.
Supports NCNN-exported models for best ARM performance.
"""

import time
import cv2
from ultralytics import YOLO

from config import (
    get_model_path, CAM_INDEX, FRAME_W, FRAME_H, TARGET_FPS,
    IMGSZ, CONF_THRESHOLD, IOU_THRESHOLD, MAX_DET, INFER_EVERY_N,
    PET_CLASSES, CAN_CLASSES, normalize_name,
)


class Detector:
    """Manages camera capture and YOLO inference."""

    def __init__(self, model_path=None):
        path = model_path or get_model_path()
        print(f"[Detector] Loading model: {path}")
        self.model = YOLO(path, task="detect")
        print("[Detector] Model loaded.")

        self.cap = None
        self.frame_count = 0
        self.last_annotated = None

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
        return True

    def release_camera(self):
        """Release the webcam."""
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
                "frame_jpeg": bytes | None,     # JPEG-encoded annotated frame
                "detected_type": "pet" | "can" | "none",
                "detected_conf": float,
                "fps": float,
            }
        """
        if not self.cap or not self.cap.isOpened():
            return {"ok": False, "frame_jpeg": None, "detected_type": "none",
                    "detected_conf": 0.0, "fps": 0.0}

        ret, frame = self.cap.read()
        if not ret:
            return {"ok": False, "frame_jpeg": None, "detected_type": "none",
                    "detected_conf": 0.0, "fps": 0.0}

        self.frame_count += 1
        detected_type = "none"
        detected_conf = 0.0

        do_infer = (self.frame_count % INFER_EVERY_N == 0)

        if do_infer:
            results = self.model.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                max_det=MAX_DET,
                verbose=False,
            )
            r0 = results[0]
            annotated = r0.plot()
            self.last_annotated = annotated

            # Find best supported detection
            if r0.boxes is not None and len(r0.boxes) > 0:
                names = r0.names
                for b in r0.boxes:
                    cls_id = int(b.cls[0].item())
                    conf = float(b.conf[0].item())
                    cls_name = normalize_name(names.get(cls_id, str(cls_id)))

                    if cls_name in PET_CLASSES and conf > detected_conf:
                        detected_type = "pet"
                        detected_conf = conf
                    elif cls_name in CAN_CLASSES and conf > detected_conf:
                        detected_type = "can"
                        detected_conf = conf
        else:
            annotated = self.last_annotated if self.last_annotated is not None else frame

        # FPS
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
