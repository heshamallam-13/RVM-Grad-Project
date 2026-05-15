#!/usr/bin/env bash
# ============================================================
# EcoVend RVM — Autostart Setup for Raspberry Pi 5
# ============================================================
# This script configures the Raspberry Pi to automatically
# launch the EcoVend touchscreen GUI on every boot.
#
# Usage:
#   cd rpi_deploy && bash setup_autostart.sh
#
# What it does:
#   1. Creates a systemd-compatible autostart .desktop entry
#   2. Creates a launcher script that activates the venv
#   3. The GUI will start in fullscreen after desktop loads
#
# To DISABLE autostart:
#   rm ~/.config/autostart/ecovend-rvm.desktop
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$SCRIPT_DIR/venv"
GUI_SCRIPT="$SCRIPT_DIR/pi_touch_gui.py"
LAUNCHER="$SCRIPT_DIR/launch_gui.sh"
AUTOSTART_DIR="$HOME/.config/autostart"
DESKTOP_FILE="$AUTOSTART_DIR/ecovend-rvm.desktop"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  EcoVend RVM — Autostart Setup${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

# --------------------------------------------------
# 1. Create launcher script
# --------------------------------------------------
echo -e "${YELLOW}[1/3] Creating launcher script...${NC}"

cat > "$LAUNCHER" << LAUNCHER_EOF
#!/usr/bin/env bash
# EcoVend RVM — Auto-generated launcher
# This script is called by the desktop autostart entry.

cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

# Wait for display to be ready
sleep 3

# Disable screen blanking / screensaver
export DISPLAY=:0
xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true

# Launch the touchscreen GUI
exec python3 "$GUI_SCRIPT"
LAUNCHER_EOF

chmod +x "$LAUNCHER"
echo -e "${GREEN}  ✅ Launcher: $LAUNCHER${NC}"

# --------------------------------------------------
# 2. Create autostart directory and .desktop entry
# --------------------------------------------------
echo -e "${YELLOW}[2/3] Creating autostart entry...${NC}"

mkdir -p "$AUTOSTART_DIR"

cat > "$DESKTOP_FILE" << DESKTOP_EOF
[Desktop Entry]
Type=Application
Name=EcoVend RVM Detection
Comment=Automatic recyclable detection on startup
Exec=$LAUNCHER
Terminal=false
X-GNOME-Autostart-enabled=true
Hidden=false
DESKTOP_EOF

echo -e "${GREEN}  ✅ Autostart: $DESKTOP_FILE${NC}"

# --------------------------------------------------
# 3. Verify
# --------------------------------------------------
echo -e "${YELLOW}[3/3] Verifying setup...${NC}"

if [ -f "$DESKTOP_FILE" ] && [ -x "$LAUNCHER" ]; then
    echo -e "${GREEN}  ✅ All good!${NC}"
else
    echo "  ❌ Something went wrong. Check permissions."
    exit 1
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  ✅ Autostart configured successfully!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  The EcoVend GUI will now launch automatically"
echo "  every time the Raspberry Pi boots into the desktop."
echo ""
echo "  To test now:   bash $LAUNCHER"
echo "  To disable:    rm $DESKTOP_FILE"
echo ""
echo -e "${GREEN}============================================================${NC}"
echo ""
