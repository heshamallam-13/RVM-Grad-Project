from pathlib import Path

p = Path("pi_touch_gui.py")
s = p.read_text()

# Add math import
s = s.replace("import threading\n", "import threading\nimport math\n")

# Add welcome call after _build_ui()
s = s.replace(
"""        # Build UI
        self._build_ui()

        # Handle window close
""",
"""        # Build UI
        self._build_ui()

        # Professional animated welcome screen
        self._show_welcome_screen()

        # Handle window close
"""
)

# Hide welcome when start is pressed
s = s.replace(
"""    def start_detection(self):
        if self.running:
            return
""",
"""    def start_detection(self):
        if hasattr(self, "welcome_frame") and self.welcome_frame.winfo_exists():
            self.welcome_frame.destroy()

        if self.running:
            return
"""
)

welcome_methods = r'''
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

'''

# Insert welcome methods before UI Layout comment
marker = "    # ------------------------------------------------------------------\n    # UI Layout\n"
s = s.replace(marker, welcome_methods + "\n" + marker)

p.write_text(s)
print("Welcome screen patch applied successfully.")
