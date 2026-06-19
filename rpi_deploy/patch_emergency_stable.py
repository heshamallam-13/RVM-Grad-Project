from pathlib import Path

p = Path("pi_touch_gui.py")
s = p.read_text()

# Replace monitor loop to stop error spam and reconnect cleanly
start = s.index("    def _serial_monitor_loop(self):")
end = s.index("    def _show_high_weight_warning(self):", start)

new_monitor = r'''    def _serial_monitor_loop(self):
        global arduino

        while self.serial_monitor_running:
            try:
                if self.processing_item:
                    time.sleep(0.2)
                    continue

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
                print("[Arduino Monitor Error - port reset]", e)

                try:
                    if arduino:
                        arduino.close()
                except:
                    pass

                arduino = None
                time.sleep(1.0)

'''

s = s[:start] + new_monitor + s[end:]

# Replace _process_item_command if exists
if "    def _process_item_command(self, cmd):" in s:
    start = s.index("    def _process_item_command(self, cmd):")
    end = s.index("    def _processing_done(self):", start)

    new_process = r'''    def _process_item_command(self, cmd):
        send_command(cmd)

        # During press motor noise, do not touch serial.
        time.sleep(6.5)

        self.root.after(0, self._processing_done)

'''
    s = s[:start] + new_process + s[end:]

p.write_text(s)
print("Emergency stability patch applied.")
