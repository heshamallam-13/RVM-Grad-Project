#!/usr/bin/env python3
"""
EcoVend RVM — Touchscreen GUI for Raspberry Pi 5
==================================================
Fullscreen tkinter application optimised for the official
7-inch Raspberry Pi touchscreen (800×480).

Uses the OpenCV DNN detector (detector.py) for fast ONNX inference.
Launches automatically on boot via setup_autostart.sh.

Usage:
    python3 pi_touch_gui.py
"""
import serial
import glob
import sys
import os
import time
import threading
import math

# ---- Auto-set DISPLAY for Raspberry Pi (fixes SSH / headless launch) ----
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"

import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from detector import Detector
from config import (
    PET_POINTS, CAN_POINTS, PET_CLASSES, CAN_CLASSES, normalize_name,
    CONF_THRESHOLD,
)

# =========================
# GUI Config
# =========================
WINDOW_TITLE = "EcoVend — RVM Detection"
BG_COLOR = "#0d1117"
ACCENT_GREEN = "#2ea043"
ACCENT_BLUE = "#1f6feb"
ACCENT_RED = "#da3633"
ACCENT_YELLOW = "#d29922"
TEXT_COLOR = "#e6edf3"
CARD_BG = "#161b22"
BUTTON_FONT = ("Helvetica", 18, "bold")
LABEL_FONT = ("Helvetica", 14)
TITLE_FONT = ("Helvetica", 20, "bold")
SCORE_FONT = ("Helvetica", 28, "bold")
NEXT_COOLDOWN_SEC = 0.6
FRAME_INTERVAL_MS = 30  # ~33 FPS GUI refresh

# =========================
# Arduino Serial
# =========================
arduino = None
serial_lock = threading.Lock()
def connect_arduino():
    global arduino
    ports = glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*")
    print("[Arduino] Ports:", ports)

    if not ports:
        print("[Arduino] Not found")
        return None

    try:
        arduino = serial.Serial(ports[0], 9600, timeout=2)
        time.sleep(3)
        print("[Arduino] Connected:", ports[0])
        return arduino
    except Exception as e:
        print("[Arduino] Connection error:", e)
        return None

def send_command(cmd):
    global arduino

    with serial_lock:
        expected = {
            "START": "ACK_START",
            "STOP": "ACK_STOP",
            "SERVO_PET": "ACK_PET_DONE",
            "SERVO_ALUMINUM": "ACK_ALUMINUM_DONE",
            "REJECT": "ACK_REJECT_DONE",
            "GET_WEIGHT": "WEIGHT_",
        }

        if arduino is None or not arduino.is_open:
            connect_arduino()

        if not (arduino and arduino.is_open):
            return []

        try:
            arduino.reset_input_buffer()
            arduino.write((cmd + "\n").encode())
            arduino.flush()

            replies = []
            target = expected.get(cmd)

            timeout_sec = 3 if cmd in ["SERVO_PET", "SERVO_ALUMINUM"] else 12

            start_time = time.time()

            while time.time() - start_time < timeout_sec:
                line = arduino.readline().decode(errors="ignore").strip()

                if not line:
                    continue

                replies.append(line)

                if target and target in line:
                    break

            print(f"[Arduino] {cmd} -> {replies}")
            return replies

        except Exception as e:
            print("[Arduino] Send error:", e)
            try:
                arduino.close()
            except:
                pass
            arduino = None
            return []
class EcoVendApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.configure(bg=BG_COLOR)

        # Fullscreen on Pi (press Escape to exit fullscreen)
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        self.root.bind("<F11>", lambda e: self.root.attributes("-fullscreen", True))

        # State
        self.total_points = 0
        self.pet_count = 0
        self.can_count = 0
        self.last_type = "none"
        self.last_conf = 0.0
        self.last_next_time = 0.0
        self.running = False
        self.high_weight_lock = False
        self.serial_monitor_running = True
        threading.Thread(target=self._serial_monitor_loop, daemon=True).start()

        # Detector
        self.detector = Detector()

        # Build UI
        self._build_ui()

        # Professional animated welcome screen
        self._show_welcome_screen()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)


    # ------------------------------------------------------------------
    # Professional Welcome Screen
    # ------------------------------------------------------------------
    def _show_welcome_screen(self):
        self.welcome_frame = tk.Frame(self.root, bg=BG_COLOR)
        self.welcome_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.welcome_canvas = tk.Canvas(
            self.welcome_frame,
            width=800,
            height=480,
            bg=BG_COLOR,
            highlightthickness=0
        )
        self.welcome_canvas.pack(fill=tk.BOTH, expand=True)

        # Background decorative shapes
        self.welcome_canvas.create_rectangle(0, 0, 800, 480, fill="#07130d", outline="")
        self.welcome_canvas.create_oval(-120, -100, 260, 260, fill="#12351f", outline="")
        self.welcome_canvas.create_oval(610, 300, 950, 650, fill="#102a44", outline="")
        self.welcome_canvas.create_text(
            400, 32,
            text="♻ EcoVend Smart Recycling",
            fill=ACCENT_GREEN,
            font=("Helvetica", 22, "bold")
        )

        # Load friendly machine image
        try:
            img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "machine.png")
            img = Image.open(img_path).convert("RGBA")
            img.thumbnail((260, 360), Image.LANCZOS)
            self.machine_photo = ImageTk.PhotoImage(img)
            self.machine_item = self.welcome_canvas.create_image(
                185, 245,
                image=self.machine_photo
            )
        except Exception as e:
            print("[Welcome] Image error:", e)
            self.machine_item = self.welcome_canvas.create_text(
                185, 245,
                text="🤖♻",
                fill=ACCENT_GREEN,
                font=("Helvetica", 80, "bold")
            )

        # Friendly text
        self.welcome_canvas.create_text(
            535, 115,
            text="Hello & Welcome 👋",
            fill=TEXT_COLOR,
            font=("Helvetica", 28, "bold")
        )
        self.welcome_canvas.create_text(
            535, 170,
            text="I am your friend,\nRVM Machine",
            fill="#c9f7d4",
            font=("Helvetica", 24, "bold"),
            justify=tk.CENTER
        )
        self.welcome_canvas.create_text(
            535, 235,
            text="Insert PET bottles or cans\nand earn green points!",
            fill="#8b949e",
            font=("Helvetica", 15, "bold"),
            justify=tk.CENTER
        )

        # Info chips
        self.welcome_canvas.create_rectangle(390, 285, 680, 335, fill=CARD_BG, outline=ACCENT_GREEN, width=2)
        self.welcome_canvas.create_text(
            535, 310,
            text="🥤 PET  +50 pts     🥫 CAN  +100 pts",
            fill=TEXT_COLOR,
            font=("Helvetica", 14, "bold")
        )

        # Big start button
        self.welcome_start_btn = tk.Button(
            self.welcome_frame,
            text="START RECYCLING ♻",
            font=("Helvetica", 20, "bold"),
            bg=ACCENT_GREEN,
            fg="white",
            activebackground="#3fb950",
            activeforeground="white",
            relief=tk.FLAT,
            padx=28,
            pady=14,
            command=self.start_detection
        )
        self.welcome_canvas.create_window(535, 390, window=self.welcome_start_btn)

        # Animated decorations
        self.wave_item = self.welcome_canvas.create_text(
            105, 95,
            text="👋",
            fill=TEXT_COLOR,
            font=("Helvetica", 34, "bold")
        )

        self.sparkles = []
        for x, y, t in [
            (710, 70, "✨"), (735, 145, "♻"), (75, 380, "🌱"),
            (650, 430, "✨"), (315, 75, "🌍"), (35, 180, "✨")
        ]:
            item = self.welcome_canvas.create_text(
                x, y, text=t, fill=ACCENT_GREEN,
                font=("Helvetica", 22, "bold")
            )
            self.sparkles.append((item, x, y))

        self.welcome_anim_t = 0
        self._animate_welcome()

    def _animate_welcome(self):
        if not hasattr(self, "welcome_frame") or not self.welcome_frame.winfo_exists():
            return

        self.welcome_anim_t += 1
        t = self.welcome_anim_t

        # Machine bounce
        y_offset = int(8 * math.sin(t / 10))
        self.welcome_canvas.coords(self.machine_item, 185, 245 + y_offset)

        # Wave emoji animation
        wave_x = 105 + int(6 * math.sin(t / 6))
        wave_y = 95 + int(4 * math.cos(t / 8))
        self.welcome_canvas.coords(self.wave_item, wave_x, wave_y)

        # Sparkle floating
        for i, (item, x, y) in enumerate(self.sparkles):
            yy = y + int(8 * math.sin((t + i * 9) / 12))
            self.welcome_canvas.coords(item, x, yy)

        # Start button glow
        if t % 20 < 10:
            self.welcome_start_btn.config(bg="#2ea043")
        else:
            self.welcome_start_btn.config(bg="#3fb950")

        self.root.after(60, self._animate_welcome)


    # ------------------------------------------------------------------
    # UI Layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        # Main container
        main = tk.Frame(self.root, bg=BG_COLOR)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ---- Top: Title ----
        title_frame = tk.Frame(main, bg=BG_COLOR)
        title_frame.pack(fill=tk.X, pady=(0, 5))

        tk.Label(
            title_frame, text="♻️  EcoVend RVM", font=TITLE_FONT,
            bg=BG_COLOR, fg=ACCENT_GREEN
        ).pack(side=tk.LEFT)

        self.fps_label = tk.Label(
            title_frame, text="FPS: --", font=LABEL_FONT,
            bg=BG_COLOR, fg=ACCENT_BLUE
        )
        self.fps_label.pack(side=tk.RIGHT)

        # ---- Center: Video + Sidebar ----
        center = tk.Frame(main, bg=BG_COLOR)
        center.pack(fill=tk.BOTH, expand=True)

        # Video canvas (left)
        self.video_label = tk.Label(
            center, bg="#000000", text="Press START",
            fg=TEXT_COLOR, font=LABEL_FONT,
            relief=tk.FLAT, borderwidth=0
        )
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Sidebar (right)
        sidebar = tk.Frame(center, bg=CARD_BG, width=220)
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Score display
        tk.Label(
            sidebar, text="POINTS", font=LABEL_FONT,
            bg=CARD_BG, fg=ACCENT_YELLOW
        ).pack(pady=(15, 0))

        self.score_label = tk.Label(
            sidebar, text="0", font=SCORE_FONT,
            bg=CARD_BG, fg=TEXT_COLOR
        )
        self.score_label.pack(pady=(0, 10))

        # Counts
        self.pet_label = tk.Label(
            sidebar, text="🥤 PET: 0", font=LABEL_FONT,
            bg=CARD_BG, fg=ACCENT_GREEN
        )
        self.pet_label.pack(pady=2)

        self.can_label = tk.Label(
            sidebar, text="🥫 CAN: 0", font=LABEL_FONT,
            bg=CARD_BG, fg=ACCENT_BLUE
        )
        self.can_label.pack(pady=2)

        # Detection status
        tk.Frame(sidebar, bg="#30363d", height=1).pack(fill=tk.X, pady=10, padx=10)

        self.det_label = tk.Label(
            sidebar, text="No Detection", font=LABEL_FONT,
            bg=CARD_BG, fg=TEXT_COLOR, wraplength=200
        )
        self.det_label.pack(pady=5)

        # Status
        self.status_label = tk.Label(
            sidebar, text="Idle", font=("Helvetica", 11),
            bg=CARD_BG, fg="#8b949e", wraplength=200
        )
        self.status_label.pack(pady=5, padx=5)

        # ---- Bottom: Buttons ----
        btn_frame = tk.Frame(main, bg=BG_COLOR)
        btn_frame.pack(fill=tk.X, pady=(5, 5))

        self.btn_start = tk.Button(
            btn_frame, text="▶  START", font=BUTTON_FONT,
            bg=ACCENT_GREEN, fg="white", activebackground="#3fb950",
            relief=tk.FLAT, padx=20, pady=12,
            command=self.start_detection
        )
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.btn_next = tk.Button(
            btn_frame, text="⏭  NEXT", font=BUTTON_FONT,
            bg=ACCENT_BLUE, fg="white", activebackground="#388bfd",
            relief=tk.FLAT, padx=20, pady=12,
            state=tk.DISABLED, command=self.next_item
        )
        self.btn_next.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.btn_finish = tk.Button(
            btn_frame, text="⏹  FINISH", font=BUTTON_FONT,
            bg=ACCENT_RED, fg="white", activebackground="#f85149",
            relief=tk.FLAT, padx=20, pady=12,
            state=tk.DISABLED, command=self.finish_session
        )
        self.btn_finish.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def start_detection(self):
        if hasattr(self, "welcome_frame") and self.welcome_frame.winfo_exists():
            self.welcome_frame.destroy()

        if self.running:
            return

        # Reset session
        self.total_points = 0
        self.pet_count = 0
        self.can_count = 0
        self.last_type = "none"
        self.last_conf = 0.0
        self._update_scoreboard()

        # Open camera
        if not self.detector.open_camera():
            self.status_label.config(text="❌ Camera failed!", fg=ACCENT_RED)
            return

        self.running = True
        threading.Thread(target=send_command, args=("START",), daemon=True).start()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL)
        self.btn_finish.config(state=tk.NORMAL)
        self.status_label.config(text="Detecting... show item", fg=ACCENT_GREEN)

        # Start frame loop
        self._frame_loop()

    def _frame_loop(self):
        """Called repeatedly via root.after() to update the GUI."""
        if not self.running:
            return

        result = self.detector.read_and_detect()

        if result["ok"] and result["frame_jpeg"] is not None:
            # Convert JPEG bytes to tkinter PhotoImage
            import io
            img_data = io.BytesIO(result["frame_jpeg"])
            pil_img = Image.open(img_data)

            # Scale to fit the video label
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            if label_w > 1 and label_h > 1:
                pil_img = pil_img.resize((label_w, label_h), Image.LANCZOS)

            self._photo = ImageTk.PhotoImage(pil_img)
            self.video_label.config(image=self._photo, text="")

            # Update detection
            if result["detected_type"] != "none":
                self.last_type = result["detected_type"]
                self.last_conf = result["detected_conf"]

            # Update FPS
            self.fps_label.config(text=f"FPS: {result['fps']:.1f}")

            # Update detection label
            if self.last_type == "pet":
                self.det_label.config(
                    text=f"🥤 PET ({self.last_conf:.0%})\n+{PET_POINTS} pts",
                    fg=ACCENT_GREEN
                )
            elif self.last_type == "can":
                self.det_label.config(
                    text=f"🥫 CAN ({self.last_conf:.0%})\n+{CAN_POINTS} pts",
                    fg=ACCENT_BLUE
                )
            else:
                self.det_label.config(text="No Detection", fg="#8b949e")

        # Schedule next frame
        self.root.after(FRAME_INTERVAL_MS, self._frame_loop)

    def next_item(self):
        if not self.running:
            return
        if self.high_weight_lock:
            self.status_label.config(text="⚠️ REMOVE HIGH WEIGHT FIRST", fg=ACCENT_RED)
            return
        now = time.time()
        if now - self.last_next_time < NEXT_COOLDOWN_SEC:
            return
        self.last_next_time = now

        if self.last_type == "pet":
            self.total_points += PET_POINTS
            self.pet_count += 1
            self.status_label.config(text=f"✅ PET +{PET_POINTS}", fg=ACCENT_GREEN)
            threading.Thread(target=send_command, args=("SERVO_PET",), daemon=True).start()
            self.last_type = "none"
            self.last_conf = 0.0
        elif self.last_type == "can":
            self.total_points += CAN_POINTS
            self.can_count += 1
            self.status_label.config(text=f"✅ CAN +{CAN_POINTS}", fg=ACCENT_BLUE)
            threading.Thread(target=send_command, args=("SERVO_ALUMINUM",), daemon=True).start()
            self.last_type = "none"
            self.last_conf = 0.0
        else:
            self.status_label.config(text="⚠️ No item detected", fg=ACCENT_YELLOW)

        self._update_scoreboard()

    def finish_session(self):
        self.running = False
        self.detector.release_camera()
        threading.Thread(target=send_command, args=("STOP",), daemon=True).start()
        msg = (
            f"Session Complete!\n\n"
            f"Total Points: {self.total_points}\n"
            f"PET: {self.pet_count}  |  CAN: {self.can_count}\n"
        )
        messagebox.showinfo("EcoVend Summary", msg)

        self.btn_start.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.DISABLED)
        self.btn_finish.config(state=tk.DISABLED)
        self.status_label.config(text="Finished. Press START.", fg="#8b949e")
        self.video_label.config(image="", text="Press START")

    def _update_scoreboard(self):
        self.score_label.config(text=str(self.total_points))
        self.pet_label.config(text=f"🥤 PET: {self.pet_count}")
        self.can_label.config(text=f"🥫 CAN: {self.can_count}")

    def _serial_monitor_loop(self):
        global arduino

        while self.serial_monitor_running:
            try:
                if arduino and arduino.is_open and arduino.in_waiting:
                    line = arduino.readline().decode(errors="ignore").strip()
                    if not line:
                        continue

                    print("[Arduino Monitor]", line)

                    if line in ["REMOVE_HIGH_WEIGHT", "AUTO_WEIGHT_HIGH"]:
                        self.high_weight_lock = True
                        self.root.after(0, self._show_high_weight_warning)

                    elif line in ["WEIGHT_CLEARED", "AUTO_RESUME_RUNNING"]:
                        self.high_weight_lock = False
                        self.root.after(0, self._clear_high_weight_warning)

                time.sleep(0.05)

            except Exception as e:
                print("[Arduino Monitor Error]", e)
                time.sleep(0.5)
    def _show_high_weight_warning(self):
        self.high_weight_lock = True
        self.btn_next.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED)

        # Create full-screen warning overlay if not already visible
        if not hasattr(self, "weight_overlay") or not self.weight_overlay.winfo_exists():
            self.weight_overlay = tk.Frame(self.root, bg="#7f1d1d")
            self.weight_overlay.place(x=0, y=0, relwidth=1, relheight=1)

            self.weight_canvas = tk.Canvas(
                self.weight_overlay,
                width=800,
                height=480,
                bg="#7f1d1d",
                highlightthickness=0
            )
            self.weight_canvas.pack(fill=tk.BOTH, expand=True)

            # Decorative emergency background
            self.weight_canvas.create_rectangle(0, 0, 800, 480, fill="#7f1d1d", outline="")
            self.weight_canvas.create_oval(-120, -100, 260, 260, fill="#991b1b", outline="")
            self.weight_canvas.create_oval(600, 300, 950, 650, fill="#450a0a", outline="")

            self.weight_warning_title = self.weight_canvas.create_text(
                400, 105,
                text="⚠ ITEM TOO HEAVY",
                fill="white",
                font=("Helvetica", 38, "bold")
            )

            self.weight_warning_subtitle = self.weight_canvas.create_text(
                400, 185,
                text="REMOVE ITEM NOW",
                fill="#fee2e2",
                font=("Helvetica", 30, "bold")
            )

            self.weight_warning_info = self.weight_canvas.create_text(
                400, 255,
                text="Maximum accepted weight: 50g",
                fill="#fecaca",
                font=("Helvetica", 20, "bold")
            )

            self.weight_warning_hint = self.weight_canvas.create_text(
                400, 335,
                text="The machine will resume automatically\nwhen the item is removed",
                fill="#ffffff",
                font=("Helvetica", 16, "bold"),
                justify=tk.CENTER
            )

            self.weight_symbols = []
            for x, y, text in [
                (90, 70, "⛔"), (710, 70, "⛔"),
                (110, 390, "⚠"), (690, 390, "⚠"),
                (400, 415, "🚫")
            ]:
                item = self.weight_canvas.create_text(
                    x, y,
                    text=text,
                    fill="#ffffff",
                    font=("Helvetica", 34, "bold")
                )
                self.weight_symbols.append((item, x, y))

            self.weight_flash_on = False
            self.weight_anim_t = 0
            self._animate_high_weight_overlay()

        self.status_label.config(text="⚠ REMOVE HIGH WEIGHT", fg=ACCENT_RED)
        self.det_label.config(text="REMOVE HIGH WEIGHT\nItem is too heavy", fg=ACCENT_RED)

    def _animate_high_weight_overlay(self):
        if not hasattr(self, "weight_overlay") or not self.weight_overlay.winfo_exists():
            return

        self.weight_anim_t += 1
        t = self.weight_anim_t

        # Flash background/title
        if t % 14 < 7:
            bg = "#7f1d1d"
            title_color = "white"
        else:
            bg = "#b91c1c"
            title_color = "#fff7ed"

        self.weight_canvas.config(bg=bg)
        self.weight_canvas.itemconfig(self.weight_warning_title, fill=title_color)

        # Shake title lightly
        shake = 6 if t % 4 < 2 else -6
        self.weight_canvas.coords(self.weight_warning_title, 400 + shake, 105)

        # Move warning symbols
        for i, (item, x, y) in enumerate(self.weight_symbols):
            yy = y + int(8 * math.sin((t + i * 10) / 8))
            self.weight_canvas.coords(item, x, yy)

        self.root.after(70, self._animate_high_weight_overlay)

    def _clear_high_weight_warning(self):
        self.high_weight_lock = False

        if hasattr(self, "weight_overlay") and self.weight_overlay.winfo_exists():
            self.weight_overlay.destroy()

        self.status_label.config(
            text="Weight OK — Continue",
            fg=ACCENT_GREEN
        )
        self.det_label.config(
            text="Ready for PET/CAN",
            fg=TEXT_COLOR
        )

        if self.running:
            self.btn_next.config(state=tk.NORMAL)
            self.btn_start.config(state=tk.DISABLED)
        else:
            self.btn_start.config(state=tk.NORMAL)

    def _on_close(self):
        self.running = False
        self.detector.release_camera()
        send_command("STOP")
        self.serial_monitor_running = False
        self.root.destroy()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()

    # Hide cursor for touchscreen kiosk mode (uncomment if needed)
    # root.config(cursor="none")

    app = EcoVendApp(root)
    root.mainloop()
