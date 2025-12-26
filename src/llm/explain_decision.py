"""
LLM-based explanation module for ADAS decisions.

This script reads the latest decision from logs/decision_log.jsonl
and uses a local LLM (via Ollama) to generate a human-readable
explanation for explainable AI (XAI) purposes.
"""

import json
import subprocess
from pathlib import Path

# -------- CONFIG --------
LOG_FILE = Path("logs/decision_log.jsonl")
OLLAMA_MODEL = "phi"
# ------------------------


def ask_llm(prompt: str) -> str:
    """
    Sends a prompt to a local LLM via Ollama and returns the response.

    Uses UTF-8 encoding with error ignoring to prevent
    Windows Unicode decoding crashes.
    """
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore"
    )
    return result.stdout.strip()


def explain_last_decision():
    """
    Loads the most recent decision log entry and requests
    a natural language explanation from the LLM.
    """
    if not LOG_FILE.exists():
        raise FileNotFoundError(
            "Decision log not found. Run test_rl_agent.py first."
        )

    with LOG_FILE.open("r", encoding="utf-8") as f:
        last_entry = json.loads(f.readlines()[-1])

    prompt = f"""
You are an ADAS safety engineer.

Explain the following autonomous driving decision in 5–6 sentences.
Do NOT introduce new scenarios.

State:
- Distance to object: {last_entry['state']['distance']} meters

Action taken: {last_entry['action']}
Reward received: {last_entry['reward']}

Sensor context:
- Radar available: {last_entry['sensors']['radar']}
- Camera available: {last_entry['sensors']['camera']}
- Fusion confidence: {last_entry['sensors']['fusion_confidence']}

Focus on:
- Why the action was chosen
- Whether it was safe
- One improvement suggestion
"""

    explanation = ask_llm(prompt)

    print("\n===== LLM EXPLANATION =====\n")
    print(explanation)


if __name__ == "__main__":
    explain_last_decision()
