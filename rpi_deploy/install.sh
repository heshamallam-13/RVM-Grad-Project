#!/usr/bin/env bash
# ============================================================
# EcoVend RVM — Automated Installation for Raspberry Pi 5
# ============================================================
# Usage:  cd rpi_deploy && bash install.sh
#
# This script will:
#   1. Install system dependencies (apt)
#   2. Create a Python virtual environment
#   3. Install all Python packages
#   4. Export the YOLO model to NCNN format (ARM-optimized)
#   5. Verify webcam access
#   6. Print run instructions
# ============================================================

set -e  # exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/venv"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  EcoVend RVM — Raspberry Pi 5 Auto-Installer${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# --------------------------------------------------
# 1. System dependencies
# --------------------------------------------------
echo -e "${YELLOW}[1/6] Installing system dependencies...${NC}"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv \
    python3-dev \
    python3-pip \
    libopencv-dev \
    libatlas-base-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    v4l-utils \
    cmake \
    git \
    libgl1 \
    libglib2.0-0

echo -e "${GREEN}  ✅ System dependencies installed${NC}"

# --------------------------------------------------
# 2. Create virtual environment
# --------------------------------------------------
echo ""
echo -e "${YELLOW}[2/6] Creating Python virtual environment...${NC}"

if [ -d "$VENV_DIR" ]; then
    echo "  Virtual environment already exists at $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}  ✅ Virtual environment created at $VENV_DIR${NC}"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "  Python: $(which python3)"
echo "  Version: $(python3 --version)"

# Upgrade pip
pip install --upgrade pip --quiet

# --------------------------------------------------
# 3. Install Python packages
# --------------------------------------------------
echo ""
echo -e "${YELLOW}[3/6] Installing Python packages (this may take several minutes)...${NC}"
pip install -r "$SCRIPT_DIR/requirements.txt" --quiet
echo -e "${GREEN}  ✅ Python packages installed${NC}"

# --------------------------------------------------
# 4. Export model to NCNN
# --------------------------------------------------
echo ""
echo -e "${YELLOW}[4/6] Exporting YOLO model to NCNN format...${NC}"

NCNN_DIR="$PROJECT_DIR/rvm_best_yolov8s_ncnn_model"
PT_FILE="$PROJECT_DIR/rvm_best_yolov8s.pt"

if [ -d "$NCNN_DIR" ]; then
    echo "  NCNN model already exported, skipping."
else
    if [ ! -f "$PT_FILE" ]; then
        echo -e "${RED}  ERROR: Model file not found: $PT_FILE${NC}"
        echo "  Please copy rvm_best_yolov8s.pt to the project root."
        exit 1
    fi
    python3 "$SCRIPT_DIR/export_model.py"
fi
echo -e "${GREEN}  ✅ NCNN model ready${NC}"

# --------------------------------------------------
# 5. Verify webcam
# --------------------------------------------------
echo ""
echo -e "${YELLOW}[5/6] Testing webcam access...${NC}"

python3 -c "
import cv2, sys
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    if ret:
        print('  ✅ Webcam OK — frame captured:', frame.shape)
    else:
        print('  ⚠️  Webcam opened but could not read frame')
else:
    print('  ⚠️  Cannot open webcam (index 0). Check connection.')
    print('  The app will still install. Connect webcam before running.')
"

# --------------------------------------------------
# 6. Done
# --------------------------------------------------
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  ✅ Installation complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  To run the detection server:"
echo ""
echo -e "    ${YELLOW}cd $SCRIPT_DIR${NC}"
echo -e "    ${YELLOW}source venv/bin/activate${NC}"
echo -e "    ${YELLOW}python3 app.py${NC}"
echo ""
echo "  Then open a browser at:  http://<your-rpi-ip>:5000"
echo ""
echo "  To run benchmarks:"
echo ""
echo -e "    ${YELLOW}python3 benchmark.py${NC}"
echo ""
echo -e "${GREEN}============================================================${NC}"
echo ""
