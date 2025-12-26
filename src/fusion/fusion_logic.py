"""
fusion_logic.py

Purpose:
--------
Fuse Radar and Camera detections into a unified object list.
This module demonstrates spatial sensor fusion using:
- Radar clustering (DBSCAN)
- Camera object detection (YOLO)
- Simple geometric gating in image space

Dataset:
--------
RadarScenes (Radar + Camera)

Fusion Strategy:
----------------
1. Cluster radar points → object centroids
2. Run YOLO on camera image
3. Project radar clusters into image X-axis
4. Associate radar clusters with camera boxes via pixel gating
5. Assign confidence based on sensor agreement
"""

import os
import h5py
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# ---------------- CONFIG ----------------
DATASET_ROOT = "RadarScenes/data"
SEQUENCE_NAME = "sequence_1"

YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.25

IMG_WIDTH = 1067          # camera image width
RADAR_GATE_PX = 60        # pixel gating for radar-camera association
# --------------------------------------


def load_data():
    """
    Load radar data and camera metadata
    """
    radar_path = os.path.join(DATASET_ROOT, SEQUENCE_NAME, "radar_data.h5")
    with h5py.File(radar_path, "r") as f:
        radar = f["radar_data"][:]
        radar_ts = np.unique(radar["timestamp"])

    cam_dir = os.path.join(DATASET_ROOT, SEQUENCE_NAME, "camera")
    cam_imgs = sorted(os.listdir(cam_dir))
    cam_ts = np.array([int(os.path.splitext(i)[0]) for i in cam_imgs])

    return radar, radar_ts, cam_imgs, cam_ts, cam_dir


def radar_clusters(frame):
    """
    Cluster radar points into object centroids using DBSCAN
    """
    X = np.column_stack((frame["x_cc"], frame["y_cc"]))

    clustering = DBSCAN(eps=2.5, min_samples=5).fit(X)
    labels = clustering.labels_

    clusters = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        pts = X[labels == lbl]
        clusters.append(pts.mean(axis=0))

    return clusters


def project_radar_to_image(y):
    """
    Project radar Y-coordinate into image X-axis (simple normalization)
    """
    norm_x = (y + 10) / 20
    return int(norm_x * IMG_WIDTH)


def main():
    radar, radar_ts, cam_imgs, cam_ts, cam_dir = load_data()
    model = YOLO(YOLO_MODEL)

    # Select a mid-sequence timestamp
    ts = radar_ts[len(radar_ts) // 2]

    # --- Radar processing ---
    frame = radar[radar["timestamp"] == ts]
    clusters = radar_clusters(frame)

    # --- Camera selection ---
    idx = np.argmin(np.abs(cam_ts - ts))
    img_path = os.path.join(cam_dir, cam_imgs[idx])
    img = cv2.imread(img_path)

    # --- Camera detection ---
    results = model.predict(
        img,
        conf=CONF_THRESH,
        imgsz=640,
        verbose=False
    )
    boxes = results[0].boxes.xyxy.cpu().numpy()

    fused_objects = []
    used_boxes = set()

    # -------- Radar-first fusion --------
    for cid, c in enumerate(clusters):
        px = project_radar_to_image(c[1])
        matched = False

        for bi, box in enumerate(boxes):
            x1, y1, x2, y2 = box

            if x1 - RADAR_GATE_PX <= px <= x2 + RADAR_GATE_PX:
                fused_objects.append({
                    "id": cid,
                    "x": float(c[0]),
                    "y": float(c[1]),
                    "source": "fused",
                    "confidence": 0.9
                })
                used_boxes.add(bi)
                matched = True
                break

        if not matched:
            fused_objects.append({
                "id": cid,
                "x": float(c[0]),
                "y": float(c[1]),
                "source": "radar",
                "confidence": 0.7
            })

    # -------- Camera-only objects --------
    for bi, box in enumerate(boxes):
        if bi not in used_boxes:
            fused_objects.append({
                "id": 1000 + bi,
                "x": None,
                "y": None,
                "source": "camera",
                "confidence": 0.3
            })

    print("\nFUSED OBJECT LIST:")
    for obj in fused_objects:
        print(obj)


if __name__ == "__main__":
    main()
