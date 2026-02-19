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
INTERVAL = 1.0
IMGSZ = 320 # Other image sizes: 640, 416 
CONF = 0.25

latest_annotated_bgr = None
latest_lock = threading.Lock()

inference_busy = False
last_inference_time = 0.0

def run_inference(frame_rgb):
    """Run YOLO on an RGB frame, store annotated frame in BGR for OpenCV display."""
    global latest_annotated_bgr, inference_busy
    try:
        results = model.predict(
            frame_rgb,          
            imgsz=IMGSZ,
            conf=CONF,
            verbose=False
        )
        annotated = results[0].plot()  # could be RGB or BGR depending on backend
        # Force annotated to BGR for cv2.imshow by converting from RGB -> BGR
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

        with latest_lock:
            latest_annotated_bgr = annotated
    except Exception as e:
        print("Inference error:", e)
    finally:
        inference_busy = False

# ----------------------------
# FPS counter (smoothed)
# ----------------------------
fps = 0.0
last_frame_t = time.time()
alpha = 0.1  # smoothing factor (0.05-0.2 nice range)

print("Press q to quit")

while True:
    frame_rgb = picam2.capture_array()

    now = time.time()

    # Kick off inference every INTERVAL seconds (non-blocking)
    if (now - last_inference_time >= INTERVAL) and not inference_busy:
        inference_busy = True
        last_inference_time = now
        threading.Thread(target=run_inference, args=(frame_rgb.copy(),), daemon=True).start()

    # Display annotated if available; otherwise display live camera
    # Convert live frame RGB -> BGR for OpenCV display
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    with latest_lock:
        display = latest_annotated_bgr if latest_annotated_bgr is not None else frame_bgr

    # Update FPS (display loop FPS)
    dt = now - last_frame_t
    last_frame_t = now
    inst_fps = (1.0 / dt) if dt > 0 else 0.0
    fps = (1 - alpha) * fps + alpha * inst_fps

    # Draw FPS bottom-right
    text = f"{fps:.1f} FPS"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    h, w = display.shape[:2]
    x = w - tw - 10
    y = h - 10
    cv2.putText(display, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("YOLO Fixed Interval (threaded)", display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
