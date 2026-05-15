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

import sys
import os
import time
import threading

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

        # Detector
        self.detector = Detector()

        # Build UI
        self._build_ui()

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

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

        now = time.time()
        if now - self.last_next_time < NEXT_COOLDOWN_SEC:
            return
        self.last_next_time = now

        if self.last_type == "pet":
            self.total_points += PET_POINTS
            self.pet_count += 1
            self.status_label.config(text=f"✅ PET +{PET_POINTS}", fg=ACCENT_GREEN)
            self.last_type = "none"
            self.last_conf = 0.0
        elif self.last_type == "can":
            self.total_points += CAN_POINTS
            self.can_count += 1
            self.status_label.config(text=f"✅ CAN +{CAN_POINTS}", fg=ACCENT_BLUE)
            self.last_type = "none"
            self.last_conf = 0.0
        else:
            self.status_label.config(text="⚠️ No item detected", fg=ACCENT_YELLOW)

        self._update_scoreboard()

    def finish_session(self):
        self.running = False
        self.detector.release_camera()

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

    def _on_close(self):
        self.running = False
        self.detector.release_camera()
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
