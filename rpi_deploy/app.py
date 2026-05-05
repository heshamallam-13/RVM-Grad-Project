"""
Flask + Socket.IO web server for RVM detection on Raspberry Pi 5.

Detection starts AUTOMATICALLY when the server launches.
Access from any browser: http://<rpi-ip>:5000
"""

import sys
import os
import time
import base64
import threading

sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
from detector import Detector
from config import HOST, PORT, PET_POINTS, CAN_POINTS

# -----------------------------------------------------------------
# App setup
# -----------------------------------------------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = "rvm-ecovend-2025"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# -----------------------------------------------------------------
# Global state
# -----------------------------------------------------------------
detector = Detector()

session = {
    "running": True,
    "total_points": 0,
    "pet_count": 0,
    "can_count": 0,
    "last_type": "none",
    "last_conf": 0.0,
    "fps": 0.0,
}

lock = threading.Lock()
NEXT_COOLDOWN = 0.6
_last_next_time = 0.0


# -----------------------------------------------------------------
# Routes
# -----------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with lock:
        return jsonify(session)


# -----------------------------------------------------------------
# Socket.IO events
# -----------------------------------------------------------------
@socketio.on("connect")
def on_connect():
    with lock:
        socketio.emit("state", session)


@socketio.on("next_item")
def on_next_item():
    global _last_next_time
    now = time.time()

    with lock:
        if not session["running"]:
            socketio.emit("toast", {"msg": "Session not active.", "type": "warn"})
            return

        if now - _last_next_time < NEXT_COOLDOWN:
            socketio.emit("toast", {"msg": "Wait a moment...", "type": "warn"})
            return
        _last_next_time = now

        if session["last_type"] == "pet":
            session["total_points"] += PET_POINTS
            session["pet_count"] += 1
            session["last_type"] = "none"
            session["last_conf"] = 0.0
            socketio.emit("toast", {"msg": f"✅ PET counted (+{PET_POINTS} pts)", "type": "ok"})
        elif session["last_type"] == "can":
            session["total_points"] += CAN_POINTS
            session["can_count"] += 1
            session["last_type"] = "none"
            session["last_conf"] = 0.0
            socketio.emit("toast", {"msg": f"✅ CAN counted (+{CAN_POINTS} pts)", "type": "ok"})
        else:
            socketio.emit("toast", {"msg": "⚠️ No item detected. Hold item in view.", "type": "warn"})

        socketio.emit("state", session)


@socketio.on("reset_session")
def on_reset():
    with lock:
        session["total_points"] = 0
        session["pet_count"] = 0
        session["can_count"] = 0
        session["last_type"] = "none"
        session["last_conf"] = 0.0
        socketio.emit("state", session)
        socketio.emit("toast", {"msg": "Session reset.", "type": "ok"})


# -----------------------------------------------------------------
# Background detection thread — starts automatically
# -----------------------------------------------------------------
def detection_loop():
    """Continuously reads frames, runs detection, streams to clients."""
    detector.open_camera()
    session["running"] = True

    while session["running"]:
        result = detector.read_and_detect()

        if not result["ok"]:
            time.sleep(0.1)
            continue

        with lock:
            session["fps"] = result["fps"]
            if result["detected_type"] != "none":
                session["last_type"] = result["detected_type"]
                session["last_conf"] = result["detected_conf"]

        # Send frame as base64 JPEG
        b64 = base64.b64encode(result["frame_jpeg"]).decode("utf-8")
        socketio.emit("frame", {"img": b64})
        socketio.emit("stats", {
            "fps": result["fps"],
            "type": session["last_type"],
            "conf": round(session["last_conf"], 2),
            "points": session["total_points"],
            "pet": session["pet_count"],
            "can": session["can_count"],
        })

        # Small sleep to avoid flooding at >30fps
        time.sleep(0.01)

    detector.release_camera()


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("  EcoVend RVM Detection — Raspberry Pi 5")
    print(f"  Open browser: http://<your-rpi-ip>:{PORT}")
    print("=" * 50)

    # Start detection in background thread
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()

    socketio.run(app, host=HOST, port=PORT, debug=False, allow_unsafe_werkzeug=True)
