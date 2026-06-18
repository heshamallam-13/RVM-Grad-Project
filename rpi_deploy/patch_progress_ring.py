from pathlib import Path

p = Path("pi_touch_gui.py")
s = p.read_text()

# Add progress target after FRAME_INTERVAL_MS
s = s.replace(
"FRAME_INTERVAL_MS = 30  # ~33 FPS GUI refresh\n",
"FRAME_INTERVAL_MS = 30  # ~33 FPS GUI refresh\nPOINTS_TARGET = 300\n"
)

# Replace score label block with canvas ring
old = '''        self.score_label = tk.Label(
            sidebar, text="0", font=SCORE_FONT,
            bg=CARD_BG, fg=TEXT_COLOR
        )
        self.score_label.pack(pady=(0, 10))
'''

new = '''        self.points_canvas = tk.Canvas(
            sidebar,
            width=150,
            height=150,
            bg=CARD_BG,
            highlightthickness=0
        )
        self.points_canvas.pack(pady=(5, 10))

        self.points_canvas.create_oval(
            18, 18, 132, 132,
            outline="#30363d",
            width=10
        )

        self.points_arc = self.points_canvas.create_arc(
            18, 18, 132, 132,
            start=90,
            extent=0,
            outline=ACCENT_GREEN,
            width=10,
            style=tk.ARC
        )

        self.score_label = self.points_canvas.create_text(
            75, 65,
            text="0",
            fill=TEXT_COLOR,
            font=("Helvetica", 28, "bold")
        )

        self.points_canvas.create_text(
            75, 100,
            text="POINTS",
            fill="#8b949e",
            font=("Helvetica", 10, "bold")
        )
'''
s = s.replace(old, new)

# Replace score_label.config text calls for canvas item
s = s.replace('self.score_label.config(text=str(end_value))', 'self.points_canvas.itemconfig(self.score_label, text=str(end_value))')
s = s.replace('self.score_label.config(text=str(value))', 'self.points_canvas.itemconfig(self.score_label, text=str(value))')
s = s.replace('self.score_label.config(text=str(self.total_points))', 'self.points_canvas.itemconfig(self.score_label, text=str(self.total_points))')

# Add update progress ring inside _update_scoreboard after points itemconfig
s = s.replace(
'''    def _update_scoreboard(self):
        self.points_canvas.itemconfig(self.score_label, text=str(self.total_points))
        self.pet_label.config(text=f"🥤 PET: {self.pet_count}")
        self.can_label.config(text=f"🥫 CAN: {self.can_count}")
''',
'''    def _update_scoreboard(self):
        self.points_canvas.itemconfig(self.score_label, text=str(self.total_points))

        progress = min(self.total_points / POINTS_TARGET, 1.0)
        extent = -360 * progress
        self.points_canvas.itemconfig(self.points_arc, extent=extent)

        if progress >= 1.0:
            self.points_canvas.itemconfig(self.points_arc, outline=ACCENT_YELLOW)
        elif progress >= 0.5:
            self.points_canvas.itemconfig(self.points_arc, outline=ACCENT_GREEN)
        else:
            self.points_canvas.itemconfig(self.points_arc, outline=ACCENT_BLUE)

        self.pet_label.config(text=f"🥤 PET: {self.pet_count}")
        self.can_label.config(text=f"🥫 CAN: {self.can_count}")
'''
)

p.write_text(s)
print("Progress ring patch applied successfully.")
