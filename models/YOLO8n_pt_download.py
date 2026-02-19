from pathlib import Path
import shutil
from ultralytics import YOLO

def download_yolo_model():
    models_dir = Path.home() / "models"
    models_dir.mkdir(exist_ok=True)

    pt_path = models_dir / "yolov8n.pt"

    if pt_path.exists():
        print("PT model already exists.")
        return

    print("Downloading yolov8n.pt...")
    YOLO("yolov8n.pt")  # triggers download to cache

    cache_path = Path.home() / ".cache" / "ultralytics" / "yolov8n.pt"

    if cache_path.exists():
        shutil.copy(cache_path, pt_path)
        print(f"Saved to {pt_path}")
    else:
        print("Download failed — cache file not found.")

if __name__ == "__main__":
    download_yolo_model()