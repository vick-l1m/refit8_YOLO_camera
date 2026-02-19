import time
import cv2
from picamera2 import Picamera2

def pick_first(d, keys):
    for k in keys:
        if k in d:
            return k
    return None

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()
time.sleep(0.2)

controls = picam2.camera_controls
print("Camera controls available:")
for k, v in controls.items():
    print(f"  {k}: {v}")

# Try to find autofocus + manual focus controls
af_mode_key = pick_first(controls, ["AfMode", "AFMode", "af_mode"])
lens_key = pick_first(controls, ["LensPosition", "Focus", "FocusPosition", "lens_position"])

# Focus step & bounds (best-effort)
focus_val = 0.0
focus_step = 0.05  # small step; adjust if your camera uses different scale
focus_min, focus_max = 0.0, 1.0

if lens_key and isinstance(controls[lens_key], tuple) and len(controls[lens_key]) >= 2:
    # Some drivers expose (min, max, default) or similar
    focus_min = float(controls[lens_key][0])
    focus_max = float(controls[lens_key][1])

# Track AF state
af_on = False

def apply_controls(new_controls: dict):
    try:
        picam2.set_controls(new_controls)
        return True
    except Exception as e:
        print("Failed to set controls:", new_controls, "error:", e)
        return False

# Start with AF on if possible
if af_mode_key:
    # libcamera often uses 0=Manual, 1=Auto, 2=Continuous (varies)
    # We'll attempt Continuous first, then Auto.
    if apply_controls({af_mode_key: 2}):
        af_on = True
    elif apply_controls({af_mode_key: 1}):
        af_on = True

print("\nKeys: a=toggle autofocus | 1=focus down | 2=focus up | q=quit\n")

while True:
    frame = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Overlay text
    status = f"AF: {'ON' if af_on else 'OFF'}"
    if lens_key:
        status += f" | {lens_key}: {focus_val:.3f}"
    cv2.putText(frame_bgr, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("ArduCam Focus Control", frame_bgr)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == ord("a"):
        if not af_mode_key:
            print("No autofocus mode control found (AfMode).")
            continue

        # Toggle AF
        if af_on:
            # Switch to Manual
            if apply_controls({af_mode_key: 0}):
                af_on = False
        else:
            # Try Continuous then Auto
            if apply_controls({af_mode_key: 2}):
                af_on = True
            elif apply_controls({af_mode_key: 1}):
                af_on = True

    if key == ord("1"):
        if not lens_key:
            print("No manual focus control found (LensPosition).")
            continue
        # Turning focus implies manual
        if af_mode_key:
            apply_controls({af_mode_key: 0})
            af_on = False

        focus_val = max(focus_min, focus_val - focus_step)
        apply_controls({lens_key: focus_val})
        print(f"{lens_key} -> {focus_val}")

    if key == ord("2"):
        if not lens_key:
            print("No manual focus control found (LensPosition).")
            continue
        if af_mode_key:
            apply_controls({af_mode_key: 0})
            af_on = False

        focus_val = min(focus_max, focus_val + focus_step)
        apply_controls({lens_key: focus_val})
        print(f"{lens_key} -> {focus_val}")

cv2.destroyAllWindows()
picam2.stop()
