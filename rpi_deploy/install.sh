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
# Install packages (with fallbacks for Bookworm+ name changes)
sudo apt-get install -y -qq \
    python3-venv \
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-opencv \
    python3-matplotlib \
    libopencv-dev \
    libhdf5-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    v4l-utils \
    cmake \
    git \
    libgl1 \
    gfortran

# libatlas-base-dev was replaced by libopenblas-dev on Bookworm+
sudo apt-get install -y -qq libatlas-base-dev 2>/dev/null || \
    sudo apt-get install -y -qq libopenblas-dev

# libglib2.0-0 was renamed to libglib2.0-0t64 on 64-bit Bookworm+
sudo apt-get install -y -qq libglib2.0-0 2>/dev/null || \
    sudo apt-get install -y -qq libglib2.0-0t64

echo -e "${GREEN}  ✅ System dependencies installed${NC}"

# --------------------------------------------------
# 2. Create virtual environment
# --------------------------------------------------
echo ""
echo -e "${YELLOW}[2/6] Creating Python virtual environment...${NC}"

# Remove old venv that may not have --system-site-packages
if [ -d "$VENV_DIR" ] && ! grep -q 'false' "$VENV_DIR/pyvenv.cfg" 2>/dev/null; then
    echo "  Virtual environment already exists at $VENV_DIR"
else
    rm -rf "$VENV_DIR"
    python3 -m venv --system-site-packages "$VENV_DIR"
    echo -e "${GREEN}  ✅ Virtual environment created at $VENV_DIR (with system packages)${NC}"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
echo "  Python: $(which python3)"
echo "  Version: $(python3 --version)"

# Upgrade pip
pip install --upgrade pip --quiet --retries 5 --timeout 60

# --------------------------------------------------
# 3. Install Python packages
# --------------------------------------------------
echo ""
echo -e "${YELLOW}[3/6] Installing Python packages (this may take several minutes)...${NC}"
pip install -r "$SCRIPT_DIR/requirements.txt" --retries 5 --timeout 120
echo -e "${GREEN}  ✅ Python packages installed${NC}"

# --------------------------------------------------
# 4. Check for ONNX model and class names
# --------------------------------------------------
echo ""
echo -e "${YELLOW}[4/6] Checking for ONNX model and class names...${NC}"

ONNX_FILE="$PROJECT_DIR/rvm_best_yolov8s.onnx"
CLASS_NAMES_FILE="$PROJECT_DIR/class_names.txt"

MISSING=0
if [ ! -f "$ONNX_FILE" ]; then
    echo -e "${RED}  ERROR: ONNX model not found at $ONNX_FILE${NC}"
    MISSING=1
fi
if [ ! -f "$CLASS_NAMES_FILE" ]; then
    echo -e "${RED}  ERROR: class_names.txt not found at $CLASS_NAMES_FILE${NC}"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "  Since PyTorch cannot be installed on a 32-bit Raspberry Pi,"
    echo "  you must export the model on your PC first:"
    echo "    1. On your PC, run: python rpi_deploy/export_model.py"
    echo "    2. Copy the generated 'rvm_best_yolov8s.onnx' and 'class_names.txt'"
    echo "       to the main project folder on the Raspberry Pi."
    echo ""
    exit 1
else
    echo -e "${GREEN}  ✅ ONNX model and class names are present${NC}"
fi

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
