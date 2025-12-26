"""
fused_tracker.py

Purpose:
--------
Defines a Kalman-filter-based track for fused sensor objects.
Each track maintains position, velocity, uncertainty, and confidence,
and is updated using fused radar-camera measurements.

Used by:
--------
- run_fused_tracking.py
- test_fused_tracking.py
"""

import numpy as np


class FusedTrack:
    """
    Represents a single tracked object using a constant-velocity
    Kalman Filter.
    """

    _id_counter = 0

    def __init__(self, x, y, confidence):
        """
        Initialize a new track.

        State vector:
            [x, y, vx, vy]

        Parameters:
        -----------
        x, y : float
            Initial position
        confidence : float
            Initial confidence score
        """
        self.id = FusedTrack._id_counter
        FusedTrack._id_counter += 1

        # State and covariance
        self.x = np.array([x, y, 0.0, 0.0], dtype=float)
        self.P = np.eye(4) * 5.0

        self.confidence = confidence
        self.missed = 0

        # Motion and observation models
        self.F = np.eye(4)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Noise models
        self.Q = np.eye(4) * 0.1    # process noise
        self.R = np.eye(2) * 1.0    # measurement noise

    def predict(self, dt=0.1):
        """
        Predict track state forward in time.
        """
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, confidence):
        """
        Update track using a new measurement.

        Parameters:
        -----------
        z : np.ndarray
            Measurement [x, y]
        confidence : float
            Confidence of the measurement
        """
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

        # Keep highest confidence
        self.confidence = max(self.confidence, confidence)
        self.missed = 0

    def pos(self):
        """
        Return current estimated position.
        """
        return self.x[0], self.x[1]
