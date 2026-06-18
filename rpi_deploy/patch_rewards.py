from pathlib import Path

p = Path("pi_touch_gui.py")
s = p.read_text()

# Add reward state after running = False area
s = s.replace(
"""self.running = False
        self.high_weight_lock""",
"""self.running = False
        self.displayed_points = 0
        self.unlocked_achievements = set()
        self.high_weight_lock"""
)

# Replace next_item
start = s.index("    def next_item(self):")
end = s.index("    def finish_session(self):", start)

new_next_and_methods = r'''    def next_item(self):
        if not self.running:
            return

        if self.high_weight_lock:
            self.status_label.config(text="⚠️ REMOVE HIGH WEIGHT FIRST", fg=ACCENT_RED)
            return

        now = time.time()
        if now - self.last_next_time < NEXT_COOLDOWN_SEC:
            return
        self.last_next_time = now

        old_points = self.total_points

        if self.last_type == "pet":
            self.total_points += PET_POINTS
            self.pet_count += 1
            self.status_label.config(text=f"✅ PET +{PET_POINTS}", fg=ACCENT_GREEN)
            self._show_reward_effect("🥤 PET DETECTED", f"+{PET_POINTS} POINTS", ACCENT_GREEN)
            threading.Thread(target=send_command, args=("SERVO_PET",), daemon=True).start()
            self.last_type = "none"
            self.last_conf = 0.0

        elif self.last_type == "can":
            self.total_points += CAN_POINTS
            self.can_count += 1
            self.status_label.config(text=f"✅ CAN +{CAN_POINTS}", fg=ACCENT_BLUE)
            self._show_reward_effect("🥫 CAN DETECTED", f"+{CAN_POINTS} POINTS", ACCENT_BLUE)
            threading.Thread(target=send_command, args=("SERVO_ALUMINUM",), daemon=True).start()
            self.last_type = "none"
            self.last_conf = 0.0

        else:
            self.status_label.config(text="⚠️ No item detected", fg=ACCENT_YELLOW)
            return

        self._animate_points(old_points, self.total_points)
        self._update_scoreboard()
        self._check_achievements()

    def _animate_points(self, start_value, end_value):
        steps = 15
        diff = end_value - start_value

        def step(i=0):
            if i >= steps:
                self.displayed_points = end_value
                self.score_label.config(text=str(end_value))
                return

            value = int(start_value + (diff * (i + 1) / steps))
            self.displayed_points = value
            self.score_label.config(text=str(value))
            self.root.after(25, lambda: step(i + 1))

        step()

    def _show_reward_effect(self, title, points_text, color):
        # Full-screen transparent-feeling overlay
        self.reward_overlay = tk.Frame(self.root, bg=BG_COLOR)
        self.reward_overlay.place(x=0, y=0, relwidth=1, relheight=1)

        self.reward_canvas = tk.Canvas(
            self.reward_overlay,
            width=800,
            height=480,
            bg=BG_COLOR,
            highlightthickness=0
        )
        self.reward_canvas.pack(fill=tk.BOTH, expand=True)

        # Glow card
        self.reward_canvas.create_rectangle(155, 130, 645, 340, fill="#0f2417", outline=color, width=4)
        self.reward_title = self.reward_canvas.create_text(
            400, 185,
            text=title,
            fill=color,
            font=("Helvetica", 32, "bold")
        )
        self.reward_points = self.reward_canvas.create_text(
            400, 250,
            text=points_text,
            fill="white",
            font=("Helvetica", 38, "bold")
        )
        self.reward_hint = self.reward_canvas.create_text(
            400, 305,
            text="Processing item...",
            fill="#8b949e",
            font=("Helvetica", 16, "bold")
        )

        # Sparkles around
        self.reward_sparkles = []
        for x, y, t in [
            (180, 150, "✨"), (625, 145, "✨"), (205, 320, "♻"),
            (595, 320, "🌱"), (400, 115, "⭐"), (400, 365, "⭐")
        ]:
            item = self.reward_canvas.create_text(
                x, y,
                text=t,
                fill=color,
                font=("Helvetica", 24, "bold")
            )
            self.reward_sparkles.append((item, x, y))

        self.reward_anim_t = 0
        self._animate_reward_effect(color)

    def _animate_reward_effect(self, color):
        if not hasattr(self, "reward_overlay") or not self.reward_overlay.winfo_exists():
            return

        self.reward_anim_t += 1
        t = self.reward_anim_t

        # Move title and points upward slightly
        y_shift = min(t * 2, 45)
        self.reward_canvas.coords(self.reward_title, 400, 185 - y_shift)
        self.reward_canvas.coords(self.reward_points, 400, 250 - y_shift)

        # Sparkles floating
        for i, (item, x, y) in enumerate(self.reward_sparkles):
            yy = y + int(10 * math.sin((t + i * 7) / 6)) - y_shift
            self.reward_canvas.coords(item, x, yy)

        # Flash effect
        if t % 10 < 5:
            self.reward_canvas.itemconfig(self.reward_points, fill="white")
        else:
            self.reward_canvas.itemconfig(self.reward_points, fill=color)

        if t > 35:
            self.reward_overlay.destroy()
            return

        self.root.after(40, lambda: self._animate_reward_effect(color))

    def _check_achievements(self):
        achievements = [
            (50, "🌱 ECO BEGINNER", "First green step unlocked!"),
            (100, "♻ ECO HERO", "You are helping the planet!"),
            (200, "🏆 ECO MASTER", "Outstanding recycling impact!")
        ]

        for threshold, title, subtitle in achievements:
            if self.total_points >= threshold and threshold not in self.unlocked_achievements:
                self.unlocked_achievements.add(threshold)
                self._show_achievement(title, subtitle)
                break

    def _show_achievement(self, title, subtitle):
        self.achievement_frame = tk.Frame(self.root, bg="#111827")
        self.achievement_frame.place(x=90, y=95, width=620, height=290)

        canvas = tk.Canvas(
            self.achievement_frame,
            width=620,
            height=290,
            bg="#111827",
            highlightthickness=0
        )
        canvas.pack(fill=tk.BOTH, expand=True)

        canvas.create_rectangle(0, 0, 620, 290, fill="#111827", outline=ACCENT_YELLOW, width=4)
        canvas.create_text(
            310, 70,
            text="ACHIEVEMENT UNLOCKED",
            fill=ACCENT_YELLOW,
            font=("Helvetica", 19, "bold")
        )
        canvas.create_text(
            310, 140,
            text=title,
            fill="white",
            font=("Helvetica", 32, "bold")
        )
        canvas.create_text(
            310, 200,
            text=subtitle,
            fill="#c9f7d4",
            font=("Helvetica", 17, "bold")
        )

        for x, y, t in [(70, 60, "🎉"), (550, 60, "🎊"), (80, 230, "✨"), (540, 230, "✨")]:
            canvas.create_text(x, y, text=t, fill=ACCENT_YELLOW, font=("Helvetica", 28, "bold"))

        self.root.after(1800, lambda: self.achievement_frame.destroy() if self.achievement_frame.winfo_exists() else None)

'''

s = s[:start] + new_next_and_methods + s[end:]

p.write_text(s)
print("Reward animations patch applied successfully.")
