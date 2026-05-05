# ♻️ EcoVend RVM — Reverse Vending Machine Detection

Real-time PET bottle and aluminum can detection using **YOLOv8s** on **Raspberry Pi 5** with webcam.

---

## 📦 Project Structure

```
RVM_Detection_Model/
├── rvm_best_yolov8s.pt              # Trained YOLOv8s model
├── Webcam&Reward_System.py          # Desktop GUI (PySide6 — for PC)
├── webcam_test.py                   # Simple webcam test
│
├── rpi_deploy/                      # ★ Raspberry Pi 5 deployment
│   ├── install.sh                   # One-command auto-installer
│   ├── requirements.txt             # Python dependencies
│   ├── config.py                    # Configuration
│   ├── detector.py                  # Detection engine (NCNN optimized)
│   ├── export_model.py              # Export .pt → NCNN
│   ├── app.py                       # Web server (auto-start detection)
│   ├── benchmark.py                 # Performance benchmarking + charts
│   ├── templates/index.html         # Web UI
│   ├── static/css/style.css         # Styling
│   └── static/js/app.js             # Client-side logic
│
└── README.md                        # This file
```

---

## 🚀 Raspberry Pi 5 Deployment (4GB RAM)

### Prerequisites

- Raspberry Pi 5 (4GB RAM) running **Raspberry Pi OS (64-bit)**
- USB webcam connected
- Internet connection (for first-time setup)

### Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/heshamallam-13/RVM-Grad-Project.git
cd RVM-Grad-Project

# 2. Run auto-installer (installs everything + exports model)
cd rpi_deploy
bash install.sh

# 3. Start the detection server
source venv/bin/activate
python3 app.py
```

Then open a browser (on the RPi or any device on the same network):

```
http://<your-rpi-ip>:5000
```

Detection starts **automatically** — no Start button needed.

---

## 📊 Benchmarking

Run the benchmark to measure FPS, latency, CPU, and RAM usage:

```bash
cd rpi_deploy
source venv/bin/activate
python3 benchmark.py
```

Outputs:
- `benchmark_results/benchmark_charts.png` — FPS, latency, CPU & RAM charts
- `benchmark_results/benchmark_data.csv` — raw data

---

## 🖥️ Web UI Features

- **Live webcam feed** with YOLO detection overlay
- **Real-time FPS** counter and chart
- **Reward system** — PET (+50 pts) / CAN (+100 pts)
- **Count Item** button to log detected items
- **Reset** to start a new session
- Dark glassmorphic responsive design

---

## ⚙️ Configuration

Edit `rpi_deploy/config.py` to adjust:

| Setting         | Default | Description                        |
|----------------|---------|------------------------------------|
| `CAM_INDEX`    | `0`     | Webcam device index                |
| `FRAME_W/H`   | 640×480 | Camera resolution                  |
| `IMGSZ`        | `640`   | YOLO inference resolution          |
| `CONF_THRESHOLD` | `0.80` | Detection confidence threshold   |
| `PET_POINTS`   | `50`    | Points per PET bottle              |
| `CAN_POINTS`   | `100`   | Points per aluminum can            |
| `PORT`         | `5000`  | Web server port                    |

---

## 🔧 Model

- **Architecture**: YOLOv8s (small)
- **Format**: NCNN (auto-exported for ARM optimization)
- **Classes**: PET bottles, Aluminum cans
- **Size**: ~22MB (.pt) / ~11MB (NCNN)

---

## 👥 Team

Graduation Project — EcoVend RVM
