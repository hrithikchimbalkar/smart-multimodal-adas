"""
Camera Object Detection on RadarScenes Dataset

This script:
- Loads camera images from a RadarScenes sequence
- Runs YOLOv8 object detection
- Saves annotated images for visualization/debugging

Purpose:
Camera-only perception baseline before sensor fusion.
"""

import os
import cv2
from ultralytics import YOLO

# ============================================================
# Configuration (safe defaults, GitHub-friendly)
# ============================================================

DATASET_ROOT = os.environ.get("RADARSCENES_ROOT", "RadarScenes/data")
SEQUENCE_NAME = os.environ.get("RADARSCENES_SEQ", "sequence_1")

OUTPUT_DIR = "demo/camera_radarscenes"
MODEL_NAME = "yolov8n.pt"

CONF_THRESH = 0.25
IMAGE_SIZE = 640
MAX_IMAGES = 20

USE_GPU = True  # set False to force CPU


# ============================================================
# Main
# ============================================================

def main():
    """
    Runs YOLOv8 object detection on RadarScenes camera images.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cam_dir = os.path.join(DATASET_ROOT, SEQUENCE_NAME, "camera")
    images = sorted(os.listdir(cam_dir))

    print(f"Found {len(images)} camera images")

    device = 0 if USE_GPU else "cpu"
    model = YOLO(MODEL_NAME)

    for idx, img_name in enumerate(images[:MAX_IMAGES]):
        img_path = os.path.join(cam_dir, img_name)

        results = model.predict(
            source=img_path,
            conf=CONF_THRESH,
            imgsz=IMAGE_SIZE,
            device=device,
            verbose=False
        )

        annotated_img = results[0].plot()
        out_path = os.path.join(OUTPUT_DIR, f"{idx:03d}.jpg")

        cv2.imwrite(out_path, annotated_img)
        print(f"[{idx}] Saved detection → {out_path}")


if __name__ == "__main__":
    main()
