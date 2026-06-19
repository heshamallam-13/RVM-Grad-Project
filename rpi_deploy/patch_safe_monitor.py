from pathlib import Path

p = Path("pi_touch_gui.py")
s = p.read_text()

start = s.index("    def _serial_monitor_loop(self):")
end = s.index("    def _show_high_weight_warning(self):", start)

new_monitor = r'''    def _serial_monitor_loop(self):
        global arduino

        while self.serial_monitor_running:
            try:
                if serial_lock.locked():
                    time.sleep(0.05)
                    continue

                if arduino and arduino.is_open and arduino.in_waiting:
                    with serial_lock:
                        line = arduino.readline().decode(errors="ignore").strip()

                    if line:
                        print("[Arduino Monitor]", line)

                        if line in ["REMOVE_HIGH_WEIGHT", "AUTO_WEIGHT_HIGH"]:
                            self.high_weight_lock = True
                            self.root.after(0, self._show_high_weight_warning)

                        elif line in ["WEIGHT_CLEARED", "AUTO_RESUME_RUNNING"]:
                            self.high_weight_lock = False
                            self.root.after(0, self._clear_high_weight_warning)

                time.sleep(0.05)

            except Exception as e:
                print("[Arduino Monitor Reset]", e)
                try:
                    if arduino:
                        arduino.close()
                except:
                    pass
                arduino = None
                time.sleep(0.8)

'''

s = s[:start] + new_monitor + s[end:]
p.write_text(s)
print("Safe monitor applied.")
