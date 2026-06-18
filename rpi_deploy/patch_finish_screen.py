from pathlib import Path

p = Path("pi_touch_gui.py")
s = p.read_text()

# Replace finish_session()
start = s.index("    def finish_session(self):")
end = s.index("    def _update_scoreboard(self):", start)

new_finish = r'''    def finish_session(self):
        self.running = False
        self.detector.release_camera()
        threading.Thread(target=send_command, args=("STOP",), daemon=True).start()

        self.btn_start.config(state=tk.NORMAL)
        self.btn_next.config(state=tk.DISABLED)
        self.btn_finish.config(state=tk.DISABLED)
        self.status_label.config(text="Finished. Press START.", fg="#8b949e")
        self.video_label.config(image="", text="Press START")

        self._show_finish_screen()

'''

s = s[:start] + new_finish + s[end:]

# Insert finish screen methods before _update_scoreboard
marker = "    def _update_scoreboard(self):"

finish_methods = r'''    def _show_finish_screen(self):
        # Full-screen summary overlay
        self.finish_frame = tk.Frame(self.root, bg="#07130d")
        self.finish_frame.place(x=0, y=0, relwidth=1, relheight=1)

        self.finish_canvas = tk.Canvas(
            self.finish_frame,
            width=800,
            height=480,
            bg="#07130d",
            highlightthickness=0
        )
        self.finish_canvas.pack(fill=tk.BOTH, expand=True)

        # Background
        self.finish_canvas.create_rectangle(0, 0, 800, 480, fill="#07130d", outline="")
        self.finish_canvas.create_oval(-120, -120, 280, 280, fill="#12351f", outline="")
        self.finish_canvas.create_oval(580, 270, 960, 640, fill="#102a44", outline="")

        self.finish_canvas.create_text(
            400, 55,
            text="🎉 THANK YOU FOR RECYCLING 🎉",
            fill=ACCENT_GREEN,
            font=("Helvetica", 24, "bold")
        )

        self.finish_canvas.create_text(
            400, 105,
            text="EcoVend Session Summary",
            fill=TEXT_COLOR,
            font=("Helvetica", 20, "bold")
        )

        # Summary cards
        self.finish_canvas.create_rectangle(75, 145, 250, 250, fill=CARD_BG, outline=ACCENT_GREEN, width=2)
        self.finish_canvas.create_text(162, 175, text="🥤 PET", fill=ACCENT_GREEN, font=("Helvetica", 18, "bold"))
        self.finish_canvas.create_text(162, 220, text=str(self.pet_count), fill="white", font=("Helvetica", 34, "bold"))

        self.finish_canvas.create_rectangle(312, 145, 487, 250, fill=CARD_BG, outline=ACCENT_BLUE, width=2)
        self.finish_canvas.create_text(400, 175, text="🥫 CAN", fill=ACCENT_BLUE, font=("Helvetica", 18, "bold"))
        self.finish_canvas.create_text(400, 220, text=str(self.can_count), fill="white", font=("Helvetica", 34, "bold"))

        self.finish_canvas.create_rectangle(550, 145, 725, 250, fill=CARD_BG, outline=ACCENT_YELLOW, width=2)
        self.finish_canvas.create_text(637, 175, text="🏆 POINTS", fill=ACCENT_YELLOW, font=("Helvetica", 18, "bold"))
        self.finish_canvas.create_text(637, 220, text=str(self.total_points), fill="white", font=("Helvetica", 34, "bold"))

        # Environmental impact estimate
        total_items = self.pet_count + self.can_count
        co2_saved = total_items * 0.12
        water_saved = total_items * 2

        self.finish_canvas.create_rectangle(110, 285, 690, 355, fill="#0f2417", outline=ACCENT_GREEN, width=2)
        self.finish_canvas.create_text(
            400, 310,
            text=f"🌍 Estimated Impact:  {co2_saved:.1f} kg CO₂ saved   •   💧 {water_saved} L water saved",
            fill="#c9f7d4",
            font=("Helvetica", 15, "bold")
        )
        self.finish_canvas.create_text(
            400, 337,
            text="Small actions create a cleaner future ♻",
            fill="#8b949e",
            font=("Helvetica", 12, "bold")
        )

        # Button
        self.finish_done_btn = tk.Button(
            self.finish_frame,
            text="START NEW SESSION ♻",
            font=("Helvetica", 17, "bold"),
            bg=ACCENT_GREEN,
            fg="white",
            activebackground="#3fb950",
            activeforeground="white",
            relief=tk.FLAT,
            padx=24,
            pady=12,
            command=self._close_finish_screen
        )
        self.finish_canvas.create_window(400, 410, window=self.finish_done_btn)

        # Confetti
        self.confetti_items = []
        symbols = ["🎉", "🎊", "✨", "♻", "🌱", "⭐"]
        colors = [ACCENT_GREEN, ACCENT_BLUE, ACCENT_YELLOW, "#ffffff", "#ff7b72"]

        for i in range(32):
            x = (i * 73) % 800
            y = -((i * 31) % 260)
            sym = symbols[i % len(symbols)]
            color = colors[i % len(colors)]
            item = self.finish_canvas.create_text(
                x, y,
                text=sym,
                fill=color,
                font=("Helvetica", 18 + (i % 4) * 3, "bold")
            )
            self.confetti_items.append([item, x, y, 2 + (i % 5)])

        self.confetti_anim_t = 0
        self._animate_confetti()

    def _animate_confetti(self):
        if not hasattr(self, "finish_frame") or not self.finish_frame.winfo_exists():
            return

        self.confetti_anim_t += 1

        for item_data in self.confetti_items:
            item, x, y, speed = item_data
            y += speed
            x += int(2 * math.sin((self.confetti_anim_t + x) / 10))

            if y > 500:
                y = -30

            item_data[1] = x
            item_data[2] = y
            self.finish_canvas.coords(item, x, y)

        self.root.after(50, self._animate_confetti)

    def _close_finish_screen(self):
        if hasattr(self, "finish_frame") and self.finish_frame.winfo_exists():
            self.finish_frame.destroy()

        # Show welcome screen again
        self._show_welcome_screen()

'''

s = s.replace(marker, finish_methods + marker)

p.write_text(s)
print("Finish screen patch applied successfully.")
