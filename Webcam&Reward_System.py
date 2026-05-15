import sys
import time
import cv2
import os

from ultralytics import YOLO

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSizePolicy,
    QMessageBox,
)

print("RUNNING FILE:", os.path.abspath(__file__))

# =========================
# Config
# =========================
MODEL_PATH = "rvm_best_yolov8s.pt"
CAM_INDEX =0

FRAME_W, FRAME_H = 640, 480
TARGET_FPS = 30

IMGSZ = 640
CONF = 0.50
IOU = 0.50
MAX_DET = 50

INFER_EVERY_N_FRAMES = 1
WINDOW_TITLE = "EcoVend - RVM YOLO GUI (Start / Next / Finish)"

VIDEO_W, VIDEO_H = 800, 450  # ثابت لمنع zoom loop

# =========================
# Points rules
# =========================
PET_POINTS = 50
CAN_POINTS = 100

# ✅ عدّل أسماء الكلاسات هنا حسب data.yaml بتاعك (names)
# مثال شائع: "pet", "can"
PET_CLASSES = {"pet", "pet_bottle", "plastic_bottle", "bottle", "plastic"}
CAN_CLASSES = {"can", "aluminum_can", "aluminium_can", "tin_can", "aluminum"}

# Debounce لزر Next عشان ما يعدّش مرتين بسرعة بالغلط
NEXT_COOLDOWN_SEC = 0.6


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def normalize_name(name: str) -> str:
    # normalize class names for matching
    return name.strip().lower().replace(" ", "_").replace("-", "_")


class VideoWorker(QThread):
    frame_ready = Signal(QImage)
    status_ready = Signal(str)
    detected_ready = Signal(str, float)  # (detected_type, confidence)

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.running = False

        # last detected info (for Next button)
        self.last_type = "none"     # "pet" | "can" | "none"
        self.last_conf = 0.0

        # UI overlay values (updated by GUI)
        self.total_points = 0
        self.pet_count = 0
        self.can_count = 0

    def set_scoreboard(self, total_points: int, pet_count: int, can_count: int):
        self.total_points = total_points
        self.pet_count = pet_count
        self.can_count = can_count

    def clear_last_detection(self):
        self.last_type = "none"
        self.last_conf = 0.0

    def run(self):
        self.running = True
        cap = open_camera()

        if not cap.isOpened():
            self.status_ready.emit("❌ Failed to open camera")
            self.running = False
            return

        prev_time = time.time()
        fps_smooth = 0.0
        alpha = 0.1

        frame_count = 0
        last_annotated = None

        self.status_ready.emit("✅ Detection running (use NEXT to count item)")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                self.status_ready.emit("❌ Camera read failed")
                break

            frame_count += 1
            do_infer = (frame_count % INFER_EVERY_N_FRAMES == 0)

            detected_type = "none"
            detected_conf = 0.0

            if do_infer:
                results = self.model.predict(
                    source=frame,
                    imgsz=IMGSZ,
                    conf=CONF,
                    iou=IOU,
                    max_det=MAX_DET,
                    verbose=False,
                )
                r0 = results[0]
                annotated = r0.plot()
                last_annotated = annotated

                # ---- Determine best supported detection (PET or CAN) ----
                # choose highest confidence box among supported classes
                if r0.boxes is not None and len(r0.boxes) > 0:
                    names = r0.names  # dict: class_id -> class_name
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

                # update last detection for Next
                self.last_type = detected_type
                self.last_conf = detected_conf
                self.detected_ready.emit(detected_type, detected_conf)

            else:
                annotated = last_annotated if last_annotated is not None else frame

            # FPS calc (smoothed)
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            instant_fps = 1 / dt if dt > 0 else 0
            fps_smooth = (1 - alpha) * fps_smooth + alpha * instant_fps

            # Overlay
            overlay = annotated.copy()
            cv2.putText(
                overlay,
                f"FPS: {fps_smooth:.1f} | Total Points: {self.total_points} | PET: {self.pet_count} | CAN: {self.can_count}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            # show last detected type
            det_text = "Detected: NONE"
            if self.last_type == "pet":
                det_text = f"Detected: PET ({self.last_conf:.2f}) -> +{PET_POINTS}"
            elif self.last_type == "can":
                det_text = f"Detected: CAN ({self.last_conf:.2f}) -> +{CAN_POINTS}"

            cv2.putText(
                overlay,
                det_text,
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Convert to QImage
            rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
            self.frame_ready.emit(qimg)

        cap.release()
        self.running = False
        self.status_ready.emit("🛑 Detection stopped")

    def stop(self):
        self.running = False
        self.wait()


class RVM_GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1050, 740)

        # Scores
        self.total_points = 0
        self.pet_count = 0
        self.can_count = 0

        self.last_next_time = 0.0

        # Load model once
        self.model = YOLO(MODEL_PATH)

        # Video label
        self.video_label = QLabel("Press START to begin detection")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background:#111; color:#ddd; border-radius:10px;")
        self.video_label.setFixedSize(VIDEO_W, VIDEO_H)
        self.video_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        # Scoreboard labels
        self.points_label = QLabel("Total Points: 0")
        self.points_label.setStyleSheet("font-size: 18px; font-weight: 600;")

        self.counts_label = QLabel("Counts -> PET: 0 | CAN: 0")
        self.counts_label.setStyleSheet("font-size: 16px;")

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-size: 15px;")

        # Buttons
        self.btn_start = QPushButton("Start")
        self.btn_next = QPushButton("Next")
        self.btn_finish = QPushButton("Finish")

        self.btn_start.clicked.connect(self.start_detection)
        self.btn_next.clicked.connect(self.next_item)
        self.btn_finish.clicked.connect(self.finish_session)

        # Initially disable next/finish until start
        self.btn_next.setEnabled(False)
        self.btn_finish.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_next)
        btn_row.addWidget(self.btn_finish)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        layout.addWidget(self.points_label)
        layout.addWidget(self.counts_label)
        layout.addWidget(self.status_label)
        layout.addLayout(btn_row)
        self.setLayout(layout)

        self.worker = None
        self.last_detected_type = "none"
        self.last_detected_conf = 0.0

    def start_detection(self):
        if self.worker and self.worker.isRunning():
            return

        # reset session
        self.total_points = 0
        self.pet_count = 0
        self.can_count = 0
        self.update_scoreboard()

        self.worker = VideoWorker(self.model)
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_ready.connect(self.update_status)
        self.worker.detected_ready.connect(self.on_detected)

        self.worker.set_scoreboard(self.total_points, self.pet_count, self.can_count)
        self.worker.start()

        self.btn_next.setEnabled(True)
        self.btn_finish.setEnabled(True)
        self.btn_start.setEnabled(False)

        self.update_status("Status: Detecting... present an item then press NEXT")

    def on_detected(self, det_type: str, conf: float):
        self.last_detected_type = det_type
        self.last_detected_conf = conf

    def next_item(self):
        if not (self.worker and self.worker.isRunning()):
            self.update_status("Status: Start first.")
            return

        # cooldown to avoid double count
        now = time.time()
        if now - self.last_next_time < NEXT_COOLDOWN_SEC:
            self.update_status("Status: Wait a moment then press NEXT again.")
            return
        self.last_next_time = now

        # decide points based on last detection
        if self.last_detected_type == "pet":
            self.total_points += PET_POINTS
            self.pet_count += 1
            self.update_status(f"✅ Counted PET (+{PET_POINTS}). Now show next item and press NEXT.")
            # clear last detection so user can present next item
            self.last_detected_type = "none"
            self.last_detected_conf = 0.0
            self.worker.clear_last_detection()

        elif self.last_detected_type == "can":
            self.total_points += CAN_POINTS
            self.can_count += 1
            self.update_status(f"✅ Counted CAN (+{CAN_POINTS}). Now show next item and press NEXT.")
            self.last_detected_type = "none"
            self.last_detected_conf = 0.0
            self.worker.clear_last_detection()

        else:
            self.update_status("⚠️ No supported item detected (PET/CAN). Hold item in view then press NEXT.")

        self.update_scoreboard()
        self.worker.set_scoreboard(self.total_points, self.pet_count, self.can_count)

    def finish_session(self):
        # stop worker
        if self.worker and self.worker.isRunning():
            self.worker.stop()

        # show summary
        msg = (
            f"Session Finished ✅\n\n"
            f"Total Points: {self.total_points}\n"
            f"PET Count: {self.pet_count}\n"
            f"CAN Count: {self.can_count}\n\n"
            f"Example: {self.pet_count} PET & {self.can_count} CAN"
        )
        QMessageBox.information(self, "EcoVend Summary", msg)

        # reset buttons
        self.btn_start.setEnabled(True)
        self.btn_next.setEnabled(False)
        self.btn_finish.setEnabled(False)

        self.update_status("Status: Finished. Press START to begin a new session.")

    def update_scoreboard(self):
        self.points_label.setText(f"Total Points: {self.total_points}")
        self.counts_label.setText(f"Counts -> PET: {self.pet_count} | CAN: {self.can_count}")

    def update_frame(self, qimg: QImage):
        pix = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(
            pix.scaled(VIDEO_W, VIDEO_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def update_status(self, text: str):
        self.status_label.setText(text)

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RVM_GUI()
    window.show()
    sys.exit(app.exec())
