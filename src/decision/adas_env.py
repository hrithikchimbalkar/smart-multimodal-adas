"""
adas_env.py

Purpose:
--------
Custom Gym environment representing a simplified ADAS decision problem.
The agent learns when to BRAKE or KEEP_SPEED based on perceived risk.

State Space:
------------
[ distance_to_object, relative_speed, lane_offset ]

Action Space:
-------------
0 -> KEEP_SPEED
1 -> BRAKE

This environment is intentionally lightweight and deterministic
to focus on decision logic rather than vehicle physics.
"""

import gym
import numpy as np
from gym import spaces


class ADASDecisionEnv(gym.Env):
    """
    Custom ADAS decision-making environment.
    """

    metadata = {"render.modes": []}

    def __init__(self):
        super().__init__()

        # Observation: [distance, relative_speed, lane_offset]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -50.0, -5.0], dtype=np.float32),
            high=np.array([200.0, 50.0, 5.0], dtype=np.float32),
            dtype=np.float32
        )

        # Actions: 0 = KEEP_SPEED, 1 = BRAKE
        self.action_space = spaces.Discrete(2)

        self.distance = None
        self.relative_speed = None
        self.lane_offset = None

    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        Compatible with Gym and Gymnasium APIs.
        """
        super().reset(seed=seed)

        self.distance = 120.0
        self.relative_speed = -5.0   # negative => approaching object
        self.lane_offset = 0.0

        obs = np.array(
            [self.distance, self.relative_speed, self.lane_offset],
            dtype=np.float32
        )

        return obs, {}

    def step(self, action):
        """
        Execute one action step.

        Parameters:
        -----------
        action : int
            0 = KEEP_SPEED
            1 = BRAKE
        """

        # --- Simple longitudinal dynamics ---
        if action == 1:      # BRAKE
            self.distance += 2.0
        else:                # KEEP_SPEED
            self.distance -= 10.0

        # --- Reward logic ---
        reward = 2 if self.distance > 15 else -100

        # Episode termination
        terminated = self.distance <= 0
        truncated = False

        obs = np.array(
            [self.distance, self.relative_speed, self.lane_offset],
            dtype=np.float32
        )

        return obs, reward, terminated, truncated, {}
