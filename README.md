# refit8_YOLO_camera
Designing a camera moule that can identify basic objects found inside of an audit. 

## Hardware
- **Raspberry Pi 5**: This project is designed to run solely on the Pi with all the datasets downloaded onto it
- **Raspberry Pi ArduCam 64mp autofocus camera**: Camera module
- **Waveshare 7inch LCD IPS 1024×600**: LCD display screen module

## Setup and installation
1. Raspberry Pi Downloads
```bash
sudo apt update
sudo apt install -y feh mpv python3-picamera2 python3-venv python3-pip
sudo apt install -y rpicam-apps
```

2. Setup the python environment for YOLO
```bash
python3 -m venv --system-site-packages ~/yolo-venv
source ~/yolo-venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python

pip install -r ~/refit8_YOLO_camera/requirements.txt
```

3. Download the YOLO Model
```bash
python3 ~/refit8_YOLO_camera/models/YOLO8n_pt_download.py
```

4. Create a folder for data storage
```bash
mkdir -p /captures/photos /captures/videos /captures/yolo
```

5. Operating inside the .venv
Once created once, you can activate the environment again with:
```bash
source ~/yolo-venv/bin/activate
```

To deactivate the .venv:
```bash
deactivate
```

# You can confirm with
which python
python -c "import sys; print(sys.executable)"
```

## Capturing Raw Data

**Still Image**
To take a still photo and display it on the LDC screen:
```bash
# Set the output
OUT=~/captures/photos/photo_$(date +%Y%m%d_%H%M%S).jpg
# Take a photo of the set size
rpicam-still -n -t 300 -o "$OUT" --width 1920 --height 1080
# Display it on the LCD screen
feh -F "$OUT"
```
```-n``` disables the preview window

**Recording**
To take a video while viewing it on a screen 
```bash
OUT=~/captures/videos/video_$(date +%Y%m%d_%H%M%S).mp4
rpicam-vid -t 10000 -o "$OUT" --width 1280 --height 720 --framerate 30
```
Takes a video for 10s - (```-t 10000 ms```)

To take the video and then play it back after
```bash
OUT=~/captures/videos/video_$(date +%Y%m%d_%H%M%S).mp4
rpicam-vid -n -t 10000 -o "$OUT" --width 1280 --height 720 --framerate 30
mpv --fs "$OUT"
```

To access this data:
```bash
# Photos
ls -lh ~/captures/photos
xdg-open ~/captures/photos/your_image.jpg

# Videos
ls -lh ~/captures/videos
mpv ~/captures/videos/your_video.mp4
```

## Capturing YOLO Data
To capture live YOLO data and then save it as a recording
```bash
python3 src/live_yolo_log.py
```
This will run until the user presses ```q```

This will output 2 tiles into ```~/captures/yolo/```:
- ```detections_YYYYMMDD_HHMMSS.json``` (per-frame list of everything it saw)
- ```summary_YYYYMMDD_HHMMSS.json``` (unique objects + counts)

## Focusing the camera
To launch the inteactive camera focus ui:
```bash
python3 src/focus_ui.py
```