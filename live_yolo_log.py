import time
import threading
from pathlib import Path

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = Path.home() / "models" / "yolov8n.pt"
model = YOLO(str(MODEL_PATH))

# ----------------------------
# Camera setup
# ----------------------------
picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (640, 480)}
    )
)

picam2.start()
time.sleep(0.2)

# ----------------------------
# Enable autofocus (best-effort)
# ----------------------------
controls = picam2.camera_controls
print("Camera controls available:", list(controls.keys()))

if "AfMode" in controls:
    # libcamera commonly: 0=manual, 1=auto, 2=continuous (varies by camera/driver)
    try:
        picam2.set_controls({"AfMode": 2})
        print("Autofocus: AfMode=2 (continuous)")
    except Exception as e:
        print("AfMode=2 failed:", e)
        try:
            picam2.set_controls({"AfMode": 1})
            print("Autofocus: AfMode=1 (auto)")
        except Exception as e2:
            print("AfMode=1 failed:", e2)
else:
    print("No AfMode control found; autofocus may not be exposed by this driver.")

# ----------------------------
# YOLO settings
# ----------------------------
INTERVAL = 1.0          # seconds between detections
IMGSZ = 640             # try 416 or 320 for more speed
CONF = 0.25

latest_annotated = None
latest_lock = threading.Lock()

inference_busy = False
last_inference_time = 0.0

def run_inference(frame_bgr):
    global latest_annotated, inference_busy

    try:
        # You can try half=True for speed (works on some setups; safe to remove if errors)
        results = model.predict(
            frame_bgr,
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False
        )
        annotated = results[0].plot()
        with latest_lock:
            latest_annotated = annotated
    except Exception as e:
        print("Inference error:", e)
    finally:
        inference_busy = False

print("Press q to quit")

while True:
    # Grab frame as fast as possible
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    now = time.time()

    # Kick off inference in the background every INTERVAL seconds
    if (now - last_inference_time >= INTERVAL) and not inference_busy:
        inference_busy = True
        last_inference_time = now
        threading.Thread(target=run_inference, args=(frame_bgr.copy(),), daemon=True).start()

    # Display latest annotated frame if available, else raw live frame
    with latest_lock:
        display = latest_annotated if latest_annotated is not None else frame_bgr

    cv2.imshow("YOLO Fixed Interval (threaded)", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
