"""
LEGACY FILE

Used for Gym environment registration via gym.make().
Not required in the final pipeline as the environment
is instantiated directly.
"""


from gym.envs.registration import register

register(
    id="ADASDecisionEnv-v0",
    entry_point="decision.adas_env:ADASDecisionEnv",
)
