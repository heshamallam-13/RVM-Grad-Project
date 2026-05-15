# ♻️ EcoVend RVM — Reverse Vending Machine Detection

Real-time PET bottle and aluminum can detection using **YOLOv8s** on **Raspberry Pi 5** with webcam and touchscreen.

---

## 📦 Project Structure

```
RVM_Detection_Model/
├── robflowbest.pt                   # ★ Primary YOLOv8s model (Roboflow trained)
├── robflowbest.onnx                 # ONNX export for Raspberry Pi
├── class_names.txt                  # Class mapping (0:can, 1:pet)
├── Webcam&Reward_System.py          # Desktop GUI (PySide6 — for PC testing)
│
├── rpi_deploy/                      # ★ Raspberry Pi 5 deployment
│   ├── install.sh                   # One-command auto-installer
│   ├── setup_autostart.sh           # Auto-launch GUI on boot
│   ├── pi_touch_gui.py              # ★ Touchscreen GUI (fullscreen, tkinter)
│   ├── requirements.txt             # Python dependencies
│   ├── config.py                    # Configuration (model paths, thresholds)
│   ├── detector.py                  # Detection engine (OpenCV DNN — ONNX)
│   ├── export_model.py              # Export .pt → ONNX (run on PC)
│   ├── app.py                       # Web server (alternative to touchscreen GUI)
│   ├── benchmark.py                 # Performance benchmarking
│   ├── templates/                   # Web UI HTML
│   └── static/                      # Web UI CSS/JS
│
├── rvm_best_yolov8s.pt              # Legacy model (kept for reference)
└── rvm_best_yolov8s.onnx            # Legacy ONNX export
```

---

## 🚀 Quick Start — Raspberry Pi 5

### 1. Clone the Repo on Your Pi

```bash
git clone https://github.com/heshamallam-13/RVM-Grad-Project.git
cd RVM-Grad-Project
```

### 2. Run the Installer

```bash
cd rpi_deploy
bash install.sh
```

This will install all system + Python dependencies and verify the webcam.

### 3. Launch the Touchscreen GUI

```bash
source venv/bin/activate
python3 pi_touch_gui.py
```

### 4. Enable Auto-Start on Boot

To make the GUI launch automatically every time the Pi boots:

```bash
bash setup_autostart.sh
```

To **disable** auto-start:

```bash
rm ~/.config/autostart/ecovend-rvm.desktop
```

---

## 🖥️ Desktop Testing (PC)

You can also test the model on your PC using the PySide6 GUI:

```bash
pip install ultralytics PySide6
python "Webcam&Reward_System.py"
```

---

## 🏗️ Model Details

| Property | Value |
|----------|-------|
| **Architecture** | YOLOv8s |
| **Model File** | `robflowbest.pt` → `robflowbest.onnx` |
| **Classes** | `can` (0), `pet` (1) |
| **Input Size** | 224×224 (Pi) / 640×640 (PC) |
| **Confidence** | 50% threshold |
| **Training** | Roboflow dataset |

---

## ⚙️ Configuration

All settings are in [`rpi_deploy/config.py`](rpi_deploy/config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `IMGSZ` | 224 | Input resolution for inference |
| `CONF_THRESHOLD` | 0.50 | Minimum confidence to count a detection |
| `IOU_THRESHOLD` | 0.50 | NMS IoU threshold |
| `INFER_EVERY_N` | 3 | Skip frames for speed (1 = every frame) |
| `PET_POINTS` | 50 | Points awarded per PET bottle |
| `CAN_POINTS` | 100 | Points awarded per aluminum can |

---

## 🎯 How It Works

1. **Start** — Camera opens and YOLO detection begins
2. **Present Item** — Hold a PET bottle or aluminum can in front of the camera
3. **Next** — Press to register the detected item and earn points
4. **Finish** — End the session and view your total score

The system uses a **reward points** model:
- 🥤 **PET bottle** = 50 points
- 🥫 **Aluminum can** = 100 points

---

## 📊 Performance

On **Raspberry Pi 5** with USB webcam and ONNX inference via OpenCV DNN:

| Metric | Value |
|--------|-------|
| Avg FPS | ~10-15 FPS |
| Inference | ~60-100 ms/frame |
| Input Resolution | 224×224 |
| Model Size | 21.5 MB (.pt) / 42.5 MB (.onnx) |

---

## 👥 Team

**MSA University — Graduation Project**

---

## 📄 License

This project is for academic purposes (MSA University Graduation Project).
