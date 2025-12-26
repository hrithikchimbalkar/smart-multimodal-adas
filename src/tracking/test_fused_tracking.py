"""
test_fused_tracking.py

Purpose:
--------
Validate temporal tracking of fused radar-camera objects.
This script integrates:
- Radar clustering
- Camera detection
- Radar-first fusion
- Track association over time

This is a system-level test to verify that objects
are tracked consistently across frames.

Not intended for training or deployment.
"""

import os
import h5py
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

from fused_tracker import FusedTrack
from run_fused_tracking import associate

# ---------------- CONFIG ----------------
DATASET_ROOT = "RadarScenes/data"
SEQUENCE_NAME = "sequence_1"

YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.25

IMG_WIDTH = 1067
RADAR_GATE_PX = 60

MAX_FRAMES = 20
# --------------------------------------


def project_radar_to_image(y):
    """
    Project radar lateral position into image X-axis
    """
    norm_x = (y + 10) / 20
    return int(norm_x * IMG_WIDTH)


def radar_clusters(frame):
    """
    Cluster radar points into object centroids
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

    return radar, radar_ts, cam_imgs, cam_ts


def main():
    radar, radar_ts, cam_imgs, cam_ts = load_data()
    model = YOLO(YOLO_MODEL)

    tracks = []

    for fi, ts in enumerate(radar_ts[:MAX_FRAMES]):
        frame = radar[radar["timestamp"] == ts]
        clusters = radar_clusters(frame)

        # --- Camera synchronization ---
        idx = np.argmin(np.abs(cam_ts - ts))
        img_path = os.path.join(
            DATASET_ROOT, SEQUENCE_NAME, "camera", cam_imgs[idx]
        )

        results = model.predict(
            img_path,
            conf=CONF_THRESH,
            imgsz=640,
            verbose=False
        )
        boxes = results[0].boxes.xyxy.cpu().numpy()

        fused = []
        used_boxes = set()

        # -------- Radar-first fusion --------
        for c in clusters:
            px = project_radar_to_image(c[1])
            matched = False

            for bi, box in enumerate(boxes):
                x1, _, x2, _ = box
                if x1 - RADAR_GATE_PX <= px <= x2 + RADAR_GATE_PX:
                    fused.append({
                        "x": float(c[0]),
                        "y": float(c[1]),
                        "confidence": 0.9
                    })
                    used_boxes.add(bi)
                    matched = True
                    break

            if not matched:
                fused.append({
                    "x": float(c[0]),
                    "y": float(c[1]),
                    "confidence": 0.7
                })

        # --- Update tracker ---
        tracks = associate(tracks, fused)

        print(f"\nFRAME {fi}")
        for t in tracks:
            x, y = t.pos()
            print(
                f" Track {t.id} | "
                f"x={x:6.2f} y={y:6.2f} "
                f"conf={t.confidence:.2f} "
                f"missed={t.missed}"
            )


if __name__ == "__main__":
    main()
