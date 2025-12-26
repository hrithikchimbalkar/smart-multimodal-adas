"""
test_rl_agent.py

Purpose:
--------
Evaluate a trained RL-based ADAS decision agent and log its decisions.
This script:
- Loads a trained DQN model
- Runs inference in the ADAS environment
- Logs each decision for LLM-based explainability

Used by:
--------
- decision_logger.py
- explain_decision.py
"""

from stable_baselines3 import DQN
from adas_env import ADASDecisionEnv
from decision_logger import log_decision


def main():
    # Initialize environment and load trained model
    env = ADASDecisionEnv()
    model = DQN.load("demo/adas_dqn_model")

    # --- Reset environment (Gym / Gymnasium compatible) ---
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    # --- Run inference ---
    for step in range(20):
        action, _ = model.predict(obs)

        step_out = env.step(action)

        # Handle Gym vs Gymnasium step API
        if len(step_out) == 5:
            obs, reward, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:
            obs, reward, done, _ = step_out

        distance = float(obs[0])
        action_name = "BRAKE" if action == 1 else "KEEP_SPEED"

        print(
            f"Step {step}: "
            f"action={action_name}, "
            f"distance={distance:.2f}, reward={reward}"
        )

        # --- Log decision for explainability ---
        log_decision(
            step=step,
            state={"distance": distance},
            action=action_name,
            reward=reward,
            sensors={
                "radar": True,
                "camera": True,
                "fusion_confidence": 0.7
            }
        )

        if done:
            print("Episode ended")
            break


if __name__ == "__main__":
    main()
