"""
RadarScenes Loader, Clustering, and Tracking Demo

This script demonstrates:
- Loading RadarScenes radar data
- Clustering radar point clouds using DBSCAN
- Tracking clusters across frames using a simple Kalman Filter
- Optional visualization for debugging and analysis

This file is intentionally self-contained for clarity.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import DBSCAN

# ============================================================
# Configuration (can be overridden via environment variables)
# ============================================================

DATASET_ROOT = os.environ.get("RADARSCENES_ROOT", "RadarScenes/data")
SEQUENCE_NAME = os.environ.get("RADARSCENES_SEQ", "sequence_1")

DBSCAN_EPS = 2.5
DBSCAN_MIN_SAMPLES = 5

MAX_ASSOCIATION_DIST = 4.0  # meters
MAX_TRACK_MISSES = 5

MAX_FRAMES = 50
FRAME_STRIDE = 5

ENABLE_PLOTTING = True


# ============================================================
# Data Loading Utilities
# ============================================================

def load_camera_image():
    """
    Loads the first camera image from the selected RadarScenes sequence.

    Returns:
        PIL.Image.Image: Loaded camera image.
    """
    cam_dir = os.path.join(DATASET_ROOT, SEQUENCE_NAME, "camera")
    images = sorted(os.listdir(cam_dir))
    img_path = os.path.join(cam_dir, images[0])

    img = Image.open(img_path)
    print(f"Loaded camera image: {img_path}, size={img.size}")
    return img


def load_radar_data():
    """
    Loads radar point cloud data from the RadarScenes HDF5 file.

    Returns:
        np.ndarray: Structured numpy array containing radar points.
    """
    radar_path = os.path.join(DATASET_ROOT, SEQUENCE_NAME, "radar_data.h5")

    with h5py.File(radar_path, "r") as f:
        radar = f["radar_data"][:]
        print("Radar dtype:", radar.dtype)
        print("Number of radar points:", radar.shape[0])

    return radar


# ============================================================
# Clustering Evaluation Utility (used during tuning)
# ============================================================

def evaluate_dbscan(X, eps, min_samples):
    """
    Evaluates DBSCAN clustering performance on a point cloud.

    Args:
        X (np.ndarray): Nx2 array of radar points.
        eps (float): DBSCAN epsilon.
        min_samples (int): DBSCAN min_samples.

    Returns:
        dict: Clustering statistics.
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    labels = clustering.labels_

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = np.sum(labels == -1) / len(labels)

    cluster_sizes = [
        np.sum(labels == lbl) for lbl in set(labels) if lbl != -1
    ]
    avg_cluster_size = np.mean(cluster_sizes) if cluster_sizes else 0

    return {
        "eps": eps,
        "min_samples": min_samples,
        "clusters": num_clusters,
        "noise_ratio": noise_ratio,
        "avg_cluster_size": avg_cluster_size
    }


# ============================================================
# Kalman Filter Track Definition
# ============================================================

class KalmanTrack:
    """
    Simple 2D Kalman Filter based object track.
    State: [x, y, vx, vy]
    """

    _id_counter = 0

    def __init__(self, x, y):
        self.id = KalmanTrack._id_counter
        KalmanTrack._id_counter += 1

        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 10.0

        self.F = np.eye(4)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        self.Q = np.eye(4) * 0.1
        self.R = np.eye(2) * 1.0

        self.age = 0
        self.missed = 0

    def predict(self, dt=0.1):
        """Predicts the next state."""
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        """Updates the track with a measurement."""
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.missed = 0

    def position(self):
        """Returns current (x, y) position."""
        return self.x[0], self.x[1]


# ============================================================
# Main Processing Loop
# ============================================================

def main():
    radar = load_radar_data()

    timestamps = radar["timestamp"]
    unique_ts = np.unique(timestamps)

    tracks = []

    for t_idx, ts in enumerate(unique_ts[:MAX_FRAMES]):

        if t_idx % FRAME_STRIDE != 0:
            continue

        frame = radar[timestamps == ts]
        X = np.column_stack((frame["x_cc"], frame["y_cc"]))

        if len(X) < DBSCAN_MIN_SAMPLES:
            continue

        # --- DBSCAN clustering ---
        clustering = DBSCAN(
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES
        ).fit(X)

        labels = clustering.labels_

        clusters = []
        for lbl in set(labels):
            if lbl == -1:
                continue
            pts = X[labels == lbl]
            clusters.append(pts.mean(axis=0))

        # --- Predict existing tracks ---
        for trk in tracks:
            trk.predict()

        # --- Associate clusters to tracks ---
        used = set()
        for trk in tracks:
            tx, ty = trk.position()
            dists = [
                np.linalg.norm(c - np.array([tx, ty]))
                for c in clusters
            ]

            if dists and min(dists) < MAX_ASSOCIATION_DIST:
                idx = int(np.argmin(dists))
                trk.update(clusters[idx])
                used.add(idx)
            else:
                trk.missed += 1

        # --- Create new tracks ---
        for i, c in enumerate(clusters):
            if i not in used:
                tracks.append(KalmanTrack(c[0], c[1]))

        # --- Remove stale tracks ---
        tracks = [
            t for t in tracks
            if t.missed < MAX_TRACK_MISSES
        ]

        print(f"Frame {t_idx}: clusters={len(clusters)}, tracks={len(tracks)}")

        # --- Visualization ---
        if ENABLE_PLOTTING:
            plt.figure(figsize=(6, 6))
            plt.scatter(X[:, 0], X[:, 1], s=3, c="gray", label="Radar Points")

            for trk in tracks:
                x, y = trk.position()
                plt.scatter(x, y, s=80, marker="x")
                plt.text(x, y, f"ID {trk.id}", color="red")

            plt.title(f"Radar Tracking @ frame {t_idx}")
            plt.xlabel("X_cc (m)")
            plt.ylabel("Y_cc (m)")
            plt.axis("equal")
            plt.grid(True)
            plt.legend()
            plt.show()
        else:
            plt.close("all")


if __name__ == "__main__":
    main()
