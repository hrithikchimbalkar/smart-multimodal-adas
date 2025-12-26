"""
run_fused_tracking.py

Purpose:
--------
Associate fused detections across frames and maintain active tracks.
This module implements a simple nearest-neighbor data association
with Kalman-filter-based prediction and update.

Used by:
--------
- test_fused_tracking.py
"""

import numpy as np
from fused_tracker import FusedTrack

# ---------------- CONFIG ----------------
MAX_DIST = 4.0      # maximum association distance (meters)
MAX_MISSES = 4      # max consecutive missed updates before deletion
# --------------------------------------


def associate(tracks, detections):
    """
    Associate detections to existing tracks and update track states.

    Parameters:
    -----------
    tracks : list[FusedTrack]
        Active tracks from previous frame
    detections : list[dict]
        Fused detections with keys: x, y, confidence

    Returns:
    --------
    list[FusedTrack]
        Updated list of active tracks
    """
    used = set()

    # --- Predict and associate existing tracks ---
    for trk in tracks:
        trk.predict()
        tx, ty = trk.pos()

        # Compute distances to all detections
        dists = [
            np.linalg.norm([d["x"] - tx, d["y"] - ty])
            if d["x"] is not None else 1e9
            for d in detections
        ]

        if dists and min(dists) < MAX_DIST:
            idx = int(np.argmin(dists))
            det = detections[idx]

            trk.update(
                np.array([det["x"], det["y"]]),
                det["confidence"]
            )
            used.add(idx)
        else:
            trk.missed += 1

    # --- Remove lost tracks ---
    tracks[:] = [t for t in tracks if t.missed < MAX_MISSES]

    # --- Create new tracks ---
    for i, det in enumerate(detections):
        if i not in used and det["x"] is not None:
            tracks.append(
                FusedTrack(det["x"], det["y"], det["confidence"])
            )

    return tracks
