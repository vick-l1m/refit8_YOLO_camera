# refit8_YOLO_camera
Designing a camera moule that can identify basic objects found inside of an audit. 

## Hardware
- **Raspberry Pi 5**: This project is designed to run solely on the Pi with all the datasets downloaded onto it
- **Waveshare 7inch LCD IPS 1024×600**: LCD display screen module

**Camera Modules**
- Raspberry Pi ArduCam 64mp autofocus camera

## Setup and installation
1. Raspberry Pi Downloads
```bash
sudo apt update
sudo apt install -y feh mpv python3-picamera2 python3-venv python3-pip
sudo apt install -y rpicam-apps
```

2. Setup the python environment for YOLO
```bash
# Create a venv
python3 -m venv --system-site-packages ~/yolo-venv
# activate
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
mkdir -p /captures/photos /captures/videos /captures/yolo/videos/data /captures/yolo/images/data /captures/yolo/measure_3d/images/data
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

**You can confirm with**
```bash
which python
python -c "import sys; print(sys.executable)"
```

6. Make each script excecutable
```bash
chmod +x still_image.sh
chmod +x record_video.sh
chmod +x run_yolo_still_image.sh
chmod +x run_yolo_live_record.sh
chmod +x run_yolo_measure_3d.sh

# follow this for any other scripts
```

## YOLO Measure Object in 3D
1. Calibrate the camera: see camera_calibration workspace
2. Run the script:
```bash
# run with defaults 
./camera_measurement/run_yolo_measure_3d.sh

# Example: custom measurements
./camera_measurement/run_yolo_measure_3d.sh --distance_m 1.7 --angle_deg 55 --depth_ratio 0.85 --class_name any
```

## Capturing Raw Data

**Still Image**
```bash
# run
./still_image.sh

# Example: take a image called "chair" and don't display
./still.sh --name chair --no-display
```

**Recording**
```bash
# run
./record_video.sh

# Example: record a video called "room 1" for 10s at 30fps and play after
./record_video.sh --name room_1 --fps 30 --time 1000 --play

# For a list of all commands
./record_video.sh -h
```

The defaults for this command are:
```bash
- FPS = 30
- TIME = 10s
- PLAY = 0 - Does not play after 
- PREVIEW = 1 - shows video while recording
```

To access this data:
```bash
# Photos
ls -lh /captures/photos
xdg-open /captures/photos/your_image.jpg

# Videos
ls -lh /captures/videos
xdg-open /captures/videos/your_video.mp4
```

## Capturing YOLO Data

**Images**
To capture a still image with YOLO data
```bash
# Run
./run_yolo_still_image.sh

# Example: image of room 1 and display after
./run_still_image_yolo.sh --name room_1 --display
```
This will output the data to:
- Photos
```/captures/yolo/images/<NAME>.jpeg```
- Json data
```/captures/yolo/images/data/<NAME>.json``` 

View the image with
```bash
xdg-open /captures/yolo/photos/your_image.jpg
```

**Videos**
To capture live YOLO data and then save it as a recording
```bash
# Run
./run_live_yolo_record.sh

# Example: video of hallway for 20s and play it back after
./run_live_yolo_record.sh --name hallway --play --time 20000

# Example: record until you press q
./run_live_yolo_record.sh --name manual_stop --time-ms 0
```
The default will record for 10s with a defualt timestand and preview on. 

This will output the data to:
- Video:
```/captures/yolo_videos/<NAME>.mp4```

- Snapshots folder (per video) - captures of each unique object:
```/captures/yolo/videos/data/<NAME>/```

- JSON log - contains a list of all objects, their instance count and confidence level:
```/captures/yolo/videos/data/<NAME>/events.json```

View the video with
```bash
xdg-open /captures/yolo/photos/your_image.jpg
```

## Changing the YOLO Model or Camera Module
Inside of each startup script, you can change the parameters 

### Camera
```bash
--backend picam2
--camera-num 0
``` 
This will use a picam on the default available port. Changing to 1 will activate the 2nd port
There is also support built in for a usb-connected camera which will run with opencv - see the startup script

**List cameras**
- Raspberry Pi connections - use ```Picamera2```
```bash
rpicam-hello --list-cameras
```
- USB connected cameras - use ```OpenCV```
```bash
ls /dev/video*
v4l2-ctl --list-devices
```

### YOLO Model
```bash
--model "$HOME/models/yolov8n.pt"
```
This will change the yolo model used. To add a new model, download and place it in the ```models``` folder. Then change this path accordingly.

### Other
- **ArduCam Focusing camera script**
To launch the inteactive camera focus ui:
```bash
python3 src/Arducam/focus_ui.py
```

