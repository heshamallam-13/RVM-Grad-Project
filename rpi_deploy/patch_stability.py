from pathlib import Path

p = Path("pi_touch_gui.py")
s = p.read_text()

# Add processing flag after self.running = False
s = s.replace(
"""        self.running = False
        self.displayed_points = 0""",
"""        self.running = False
        self.processing_item = False
        self.displayed_points = 0"""
)

# Replace next_item
start = s.index("    def next_item(self):")
end = s.index("    def finish_session(self):", start)

new_next = r'''    def next_item(self):
        if not self.running:
            return

        if self.processing_item:
            self.status_label.config(text="⏳ Processing... please wait", fg=ACCENT_YELLOW)
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
            self.processing_item = True
            self.btn_next.config(state=tk.DISABLED)

            self.status_label.config(text=f"✅ PET +{PET_POINTS} | Processing...", fg=ACCENT_GREEN)
            self._show_reward_effect("🥤 PET DETECTED", f"+{PET_POINTS} POINTS", ACCENT_GREEN)

            threading.Thread(
                target=self._process_item_command,
                args=("SERVO_PET",),
                daemon=True
            ).start()

            self.last_type = "none"
            self.last_conf = 0.0

        elif self.last_type == "can":
            self.total_points += CAN_POINTS
            self.can_count += 1
            self.processing_item = True
            self.btn_next.config(state=tk.DISABLED)

            self.status_label.config(text=f"✅ CAN +{CAN_POINTS} | Processing...", fg=ACCENT_BLUE)
            self._show_reward_effect("🥫 CAN DETECTED", f"+{CAN_POINTS} POINTS", ACCENT_BLUE)

            threading.Thread(
                target=self._process_item_command,
                args=("SERVO_ALUMINUM",),
                daemon=True
            ).start()

            self.last_type = "none"
            self.last_conf = 0.0

        else:
            self.status_label.config(text="⚠️ No item detected", fg=ACCENT_YELLOW)
            return

        self._animate_points(old_points, self.total_points)
        self._update_scoreboard()
        self._check_achievements()

    def _process_item_command(self, cmd):
        send_command(cmd)

        # Arduino press cycle is about 5 seconds + servo delays
        time.sleep(6.2)

        self.root.after(0, self._processing_done)

    def _processing_done(self):
        self.processing_item = False

        if self.running and not self.high_weight_lock:
            self.btn_next.config(state=tk.NORMAL)
            self.status_label.config(text="Ready for next item", fg=ACCENT_GREEN)

'''

s = s[:start] + new_next + s[end:]

# Replace serial monitor loop
start = s.index("    def _serial_monitor_loop(self):")
end = s.index("    def _show_high_weight_warning(self):", start)

new_monitor = r'''    def _serial_monitor_loop(self):
        global arduino

        while self.serial_monitor_running:
            try:
                # Do not read while a command is being sent
                if serial_lock.locked():
                    time.sleep(0.05)
                    continue

                if arduino and arduino.is_open and arduino.in_waiting:
                    with serial_lock:
                        line = arduino.readline().decode(errors="ignore").strip()

                    if not line:
                        time.sleep(0.05)
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

                try:
                    if arduino:
                        arduino.close()
                except:
                    pass

                arduino = None
                time.sleep(0.5)

'''

s = s[:start] + new_monitor + s[end:]

p.write_text(s)
print("Stability patch applied successfully.")
