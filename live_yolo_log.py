import time
from pathlib import Path
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = Path.home() / "models" / "yolov8n.pt"
model = YOLO(str(MODEL_PATH))

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(
    main={"format": "RGB888", "size": (640, 480)}
))
picam2.start()

INTERVAL = 0.5  # seconds between detections
last_inference_time = 0
latest_results = None

print("Press q to quit")

while True:
    frame = picam2.capture_array()
    now = time.time()

    # Run YOLO only every 0.5 seconds
    if now - last_inference_time >= INTERVAL:
        latest_results = model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            verbose=False
        )
        last_inference_time = now

    # If we have results, draw them
    if latest_results is not None:
        annotated = latest_results[0].plot()
    else:
        annotated = frame

    cv2.imshow("YOLO Fixed Interval (0.5s)", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
