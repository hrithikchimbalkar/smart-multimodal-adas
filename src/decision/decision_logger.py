"""
decision_logger.py

Purpose:
--------
Log high-level ADAS decision-making steps in a structured format.
These logs are later consumed by the LLM explainability module
to generate human-readable safety explanations.

Log Format:
-----------
Each line is a JSON object (JSONL), enabling easy streaming,
parsing, and retrieval.

Used by:
--------
- test_rl_agent.py
- explain_decision.py
"""

import json
import os
from datetime import datetime

# ---------------- CONFIG ----------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "decision_log.jsonl")
# --------------------------------------


def log_decision(step, state, action, reward, sensors):
    """
    Append a decision entry to the decision log.

    Parameters:
    -----------
    step : int
        Time step index
    state : dict
        Environment state (e.g., distance to object)
    action : str
        Action taken by the agent (e.g., BRAKE, KEEP_SPEED)
    reward : float
        Reward received after taking the action
    sensors : dict
        Sensor availability and fusion confidence
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "state": state,
        "action": action,
        "reward": reward,
        "sensors": sensors
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
