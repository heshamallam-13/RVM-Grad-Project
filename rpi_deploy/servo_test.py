import serial, time, glob

port = glob.glob('/dev/ttyACM*')[0]
arduino = serial.Serial(port, 9600, timeout=2)
time.sleep(3)

arduino.write(b"SERVO_PET\n")
time.sleep(10)

while arduino.in_waiting:
    print(arduino.readline().decode(errors="ignore").strip())
