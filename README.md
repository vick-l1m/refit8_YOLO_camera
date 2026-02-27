# refit8_YOLO_camera
Designing a camera moule that can identify basic objects found inside of an audit. 

## Hardware
- **Raspberry Pi 5**: This project is designed to run solely on the Pi with all the datasets downloaded onto it
- **Waveshare 7inch LCD IPS 1024×600**: LCD display screen module

**Camera Modules**
- Raspberry Pi ArduCam 64mp autofocus camera

## Setup and installation
### 1. Raspberry Pi Downloads
```bash
sudo apt update
sudo apt install -y feh mpv python3-picamera2 python3-venv python3-pip
sudo apt install -y rpicam-apps
```

### 2. Setup the python environment for YOLO
```bash
# Create a venv
python3 -m venv --system-site-packages ~/yolo-venv
# activate
source ~/yolo-venv/bin/activate
pip install --upgrade pip
pip install ultralytics opencv-python

pip install -r requirements.txt
```

### 3. Download the YOLO Model
```bash
cd models
python3 YOLO8n_pt_download.py
```

### 4. Operating inside the .venv
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

### 5. Make each script excecutable
```bash
cd scripts
chmod +x still_image.sh
chmod +x record_video.sh
chmod +x run_yolo_still_image.sh
chmod +x run_yolo_live_record.sh
chmod +x run_yolo_measure_3d.sh

# follow this for any other scripts
```

## YOLO Measure Object in 3D
Ensure you are in the .venv

### 1. Calibrate the camera: 
See camera_calibration workspace
Ensure that the intrinsics file is being used by the startup script - or run it manually as shown below

### 2. Run the script:
```bash
# run with defaults 
./run_yolo_measure_3d.sh

# Example with fake intrinsics and custom measurements
./run_yolo_measure_3d.sh  --name chair_image --intrinsics calibration/fake_intrinsics_1920x1080.json --angle_deg 45 --depth_ratio 0.85 --class_name chair --preview
```

The important parameters to change are:
- ```--name```: name of the output file. Default - ```timestamp``` 
- ```--intrinsics```: intrinsics file for the camera calibration. Default - ```captures/fake_intrinsices_1920x1080.json``` 
- ```--angle_deg```: The angle that the photo is being taken relative to the object's front face (degrees). Default - ```45``` 
- ```--depth ratio```: Estimated ratio between the width and depth - Default - ```1``` 
- ```--distance```: The distance from the camera to the object (meters). Default - ```2.0``` 
- --```class_name```: If you want to look specifically for an object type. Default - ```any```             

### Accessing data
This module will output to the folder:
```bash
refit8_YOLO_camera/captures/yolo/measure_3d/
```

- **Images** 
```bash
cd ~/refit8_YOLO_camera/captures/yolo/measure_3d/images

# Open
xdg-open <image_name>.jpeg
```

- **Data**
It will output the recognised images, parameters used and measurements into a json file. Access them here using any text editing tool:
```bash
cd ~/refit8_YOLO_camera/captures/yolo/measure_3d/data/<image_name>.json
```

### Limitations
- Reliance on YOLO
The program relies on YOLO to identify what object it is looking at so that it can create the bounding box. If the object does not exist in this YOLO model, it won't be able to measure it

- Depth calculation
Currently, the depth is calculated using a depth ratio and the angle_deg. It is doing an estimation of the depth, rather than actually measuring it

### Next steps
- Correctly doing the camera calibration and producing a ```intrinsics.json``` file
- Adding/finding a way to measure the distance from the object instead of using a constant distance
- Editing/adjusting the object detection/YOLO model so that it can identify more objects
- Adding a "known object" for calibration purposes to increase the accuracy

## Capturing Raw Data (Raspberry Pi)

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

# Example: Live log no recording, run till press q
./run_yolo_live_record.sh --no-record --time-ms 0

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
These commands work with raspberry pi and usb/opencv devices (eg a webcam)

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
The camera is automatically detected and defaults to raspberry pi. It falls back to an opencv/usb camera if a raspberry pi camera is not detected.

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

