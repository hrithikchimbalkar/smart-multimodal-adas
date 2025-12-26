"""
temporal_alignment.py

Purpose:
--------
Align Radar and Camera data temporally using timestamps.
This module demonstrates how asynchronous automotive sensors
can be synchronized without hardware triggering.

Dataset:
--------
RadarScenes (Radar + Camera)

Approach:
---------
For each radar timestamp, find the closest camera frame
based on absolute timestamp difference.

This alignment is later used by the fusion module.
"""

import os
import h5py
import numpy as np

# ---------------- CONFIG ----------------
DATASET_ROOT = "RadarScenes/data"
SEQUENCE_NAME = "sequence_1"   # keep small for now
# ---------------------------------------


def load_radar_timestamps():
    """
    Load unique radar timestamps from radar_data.h5
    """
    radar_path = os.path.join(DATASET_ROOT, SEQUENCE_NAME, "radar_data.h5")

    with h5py.File(radar_path, "r") as f:
        radar = f["radar_data"][:]
        timestamps = np.unique(radar["timestamp"])

    return timestamps


def load_camera_timestamps():
    """
    Load camera timestamps from image filenames
    """
    cam_dir = os.path.join(DATASET_ROOT, SEQUENCE_NAME, "camera")
    images = sorted(os.listdir(cam_dir))

    cam_ts = np.array([
        int(os.path.splitext(img)[0]) for img in images
    ])

    return cam_ts, images


def main():
    radar_ts = load_radar_timestamps()
    cam_ts, cam_imgs = load_camera_timestamps()

    print(f"Radar frames: {len(radar_ts)}")
    print(f"Camera frames: {len(cam_ts)}")

    print("\nSample radar → camera alignment:")

    for i in range(min(5, len(radar_ts))):
        rts = radar_ts[i]

        # Find closest camera timestamp
        idx = np.argmin(np.abs(cam_ts - rts))
        cts = cam_ts[idx]
        img = cam_imgs[idx]

        delta_ms = abs(cts - rts) / 1e6  # ns → ms

        print(
            f"Radar ts={rts}  →  "
            f"Camera img={img}  "
            f"(Δt ≈ {delta_ms:.2f} ms)"
        )


if __name__ == "__main__":
    main()
