"""
train_rl_agent.py

Purpose:
--------
Train a Reinforcement Learning (DQN) agent for ADAS decision-making.
The agent learns when to BRAKE or KEEP_SPEED based on the perceived
distance to obstacles.

Environment:
------------
ADASDecisionEnv (custom Gym environment)

Algorithm:
----------
Deep Q-Network (DQN) from Stable-Baselines3

Notes:
------
- Runs entirely on CPU (GPU not required)
- Lightweight environment suitable for laptops
"""

import gym
from stable_baselines3 import DQN
from adas_env import ADASDecisionEnv


def main():
    # Initialize custom ADAS environment
    env = ADASDecisionEnv()

    # Create DQN agent
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        buffer_size=10_000,
        learning_starts=1_000,
        batch_size=32,
        gamma=0.99,
        verbose=1,
        device="cpu"   # keep CPU-only for portability
    )

    # Train the agent
    model.learn(total_timesteps=20_000)

    # Save trained model
    model.save("demo/adas_dqn_model")

    print("RL agent training complete.")


if __name__ == "__main__":
    main()
