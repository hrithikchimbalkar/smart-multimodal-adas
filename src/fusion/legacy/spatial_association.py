"""
LEGACY FILE

Used for early visualization and sanity-check of
radar-to-camera spatial association.

Not used in final fusion pipeline.
"""



import os
import h5py
import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.cluster import DBSCAN

# -------- CONFIG --------
DATASET_ROOT = "RadarScenes/data"
SEQUENCE_NAME = "sequence_1"
YOLO_MODEL = "yolov8n.pt"
CONF_THRESH = 0.25
IMG_WIDTH = 1067   # from earlier inspection
# ------------------------

def load_radar_frame(ts):
    radar_path = os.path.join(DATASET_ROOT, SEQUENCE_NAME, "radar_data.h5")
    with h5py.File(radar_path, "r") as f:
        radar = f["radar_data"][:]
    frame = radar[radar["timestamp"] == ts]
    return frame

def radar_clusters(frame):
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

def project_radar_to_image(x, y):
    """
    Simple projection:
    map lateral position to image X axis
    """
    norm_x = (y + 10) / 20     # assume +/-10 m FOV
    px = int(norm_x * IMG_WIDTH)
    return px

def main():
    # --- Load timestamps ---
    radar_path = os.path.join(DATASET_ROOT, SEQUENCE_NAME, "radar_data.h5")
    with h5py.File(radar_path, "r") as f:
        radar = f["radar_data"][:]
        radar_ts = np.unique(radar["timestamp"])

    # Pick one timestamp
    ts = radar_ts[len(radar_ts) // 2]

    # --- Load matching camera image ---
    cam_dir = os.path.join(DATASET_ROOT, SEQUENCE_NAME, "camera")
    cam_imgs = sorted(os.listdir(cam_dir))
    cam_ts = np.array([int(os.path.splitext(i)[0]) for i in cam_imgs])
    idx = np.argmin(np.abs(cam_ts - ts))
    img_path = os.path.join(cam_dir, cam_imgs[idx])

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    # --- Radar processing ---
    frame = load_radar_frame(ts)
    clusters = radar_clusters(frame)

    # --- Camera detection ---
    model = YOLO(YOLO_MODEL)
    results = model.predict(img, conf=CONF_THRESH, imgsz=640, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    # --- Visualization ---
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for c in clusters:
        px = project_radar_to_image(c[0], c[1])
        cv2.circle(img, (px, h // 2), 6, (0, 0, 255), -1)

    cv2.imshow("Radar–Camera Spatial Association", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
