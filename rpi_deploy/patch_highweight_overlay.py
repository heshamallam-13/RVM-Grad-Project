from pathlib import Path

p = Path("pi_touch_gui.py")
s = p.read_text()

# Add high weight overlay state after high_weight_lock init
s = s.replace(
"""self.high_weight_lock = False
self.serial_monitor_running = True""",
"""self.high_weight_lock = False
self.high_weight_flash_on = False
self.serial_monitor_running = True"""
)

# Replace _show_high_weight_warning function
start = s.index("    def _show_high_weight_warning(self):")
end = s.index("    def _clear_high_weight_warning(self):", start)

new_show = r'''    def _show_high_weight_warning(self):
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

'''

s = s[:start] + new_show + s[end:]

# Replace _clear_high_weight_warning function
start = s.index("    def _clear_high_weight_warning(self):")
end = s.index("    def _on_close(self):", start)

new_clear = r'''    def _clear_high_weight_warning(self):
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

'''

s = s[:start] + new_clear + s[end:]

p.write_text(s)
print("High weight overlay patch applied successfully.")
