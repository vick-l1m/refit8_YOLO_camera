# ArduCam Setup and Installation Instructions
This doccument describes the installation of the ArduCam driver and software for Raspberry Pi 5 with the ArduCam 64 MP autofocus camera module (often called the Hawkeye)

## 1. Hardware
Power off your Pi 5 and connect the camera flex ribbon to the CSI-2 camera port (usually CAM1 on Pi 5).
Ensure the silver contacts are facing the correct direction and the connector is fully seated.

## 2. Download the ArduCam Install Script
Open a terminal on the Pi and run:
```bash
wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
chmod +x install_pivariety_pkgs.sh
```
This script helps install camera drivers and software.

## 3. Install Required Software
```bash
./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps
```
These install the libcamera framework and apps with ArduCam support (improves autofocus and sensor support).

## 4. Enable the Camera in Config
Edit the firmware configuration:
```bash
sudo nano /boot/firmware/config.txt
```
Under the [all] section, add:
```bash
dtoverlay=arducam-64mp
```
If the camera is on the cam0 port instead of cam1:
```bash
dtoverlay=arducam-64mp,cam0
```
Save and exit (Ctrl+O, Enter, Ctrl+X) then reboot:
```bash
sudo reboot
```
This tells the Raspberry Pi kernel to load the 64 MP ArduCam driver overlay.

## 5. Test the Camera
Once rebooted, try capturing an image:
```bash
rpicam-still -t 5000
```
You can also check if the camera is working with
```bash
rpicam-hello --list-devices
```

### Notes and tips
If you ever get “No cameras available!” from libcamera, re-check:
- Ribbon cable seating
- That the overlay is spelled exactly arducam-64mp in config.txt
- Reboot after edits