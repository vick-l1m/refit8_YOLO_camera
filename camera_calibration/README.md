# Camera Calibration Process

## Goal of calibration
We want to compute:
- ```fx, fy``` → focal length in pixels
- ```cx, cy``` → optical center
- ```distortion coefficients``` → lens distortion correction

These let us convert: ```pixel width  →  meters   (given distance Z)```

## 1. Print Calibration Board
Find an appropriate size here: [calibration-checkerboard-collection](https://markhedleyjones.com/projects/calibration-checkerboard-collection)

In this project, we used:
- 8 columns × 6 rows of INNER corners
- Square size: 35 mm (0.035 m)

## Step 2 — Camera Setup Rules 
You must calibrate using the exact same configuration you will use for measurement:
- Resolution: 1920×1080
- Same autofocus mode (or fixed focus)
- Same lens
- Same cropping mode
- Same backend (rpicam-still)
- Same aspect ratio
- If you change resolution later → you must recalibrate.

## Step 3 — Taking Calibration Images
Take 25–40 images with these rules:
1. Fill the frame sometimes
- Checkerboard covers most of the image
2. Make it small sometimes
- Board only takes up 30–50% of frame
3. Rotate: Take images where the board is:
- tilted left/right
- tilted up/down
- rotated diagonally
- almost touching corners of frame
4. Move it around the image
- top-left
- top-right
- bottom-left
- bottom-right
- center
5. Vary distance
- Close (~0.5 m)
- Medium (~1–1.5 m)
- Far (~2 m)

### What NOT to do
- Don’t take 30 images from the same position.
- Don’t keep it perfectly flat and centered.
- Don’t use blurry images.
- Don’t let it go out of focus.
- Don’t bend the paper.

### Take the images
use the automated script:
```bash
./run_capture_calibration_images.sh
```
This will take a photo every 3s and save them to the folder:
```bash
cv/yolo/calibration/arducam_1920x1080/
```

## Step 4 - Run Calibration Script
Once you have the images, run:
```bash 
./run_calibrate_camera.sh
```

## Step 5 - Check Calibration Quality
After running, you'll see:
```bash
Reprojection RMSE (px): 0.32
```

Good RMSE values:
- < 0.5 px → Excellent
- 0.5 – 1.0 px → Acceptable
- 1.5 px → Probably bad images

## Important
Calibrate:
- Once per camera module
- Once per resolution
- Once per lens
If you:
- change resolution
- crop
- switch Arducam module
- change focus type
→ recalibrate.