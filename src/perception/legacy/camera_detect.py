"""
LEGACY FILE
Used only for early YOLO sanity checks.
Not part of the final ADAS pipeline.
"""


import os
from ultralytics import YOLO
import urllib.request

MODEL_PATH = "yolov8n.pt"  # will download if not present
SAMPLE_IMG = "data/sample.jpg"
OUT_IMG = "demo/sample_result.jpg"

def ensure_model():
    # Ultralytics will auto-download yolov8n.pt if not present when you create YOLO("yolov8n.pt")
    pass

def run_inference():
    model = YOLO(MODEL_PATH)
    source = "data/coco_sample"   # folder instead of single image
    results = model.predict(
        source=source,
        conf=0.25,
        imgsz=640,
        save=True,
        project="demo",
        name="coco_batch",
        exist_ok=True
    )
    print(f"Processed {len(results)} images")


if __name__ == "__main__":
    run_inference()