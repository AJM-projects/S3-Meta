import gymnasium as gym
from gymnasium import spaces
import numpy as np


class MultiArmedBanditEnv(gym.Env):
    """Simple Multi-Armed Bandit environment."""

    def __init__(self, n_bandits=5):
        super().__init__()
        self.n_bandits = n_bandits

        # Action space: discrete choice of which bandit to pull
        self.action_space = spaces.Discrete(n_bandits)

        # Observation space: one-hot encoding of last action taken
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(n_bandits,), dtype=np.float32
        )

        self.last_action = -1

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.last_action = -1

        # Initial observation is all zeros (no action taken yet)
        obs = np.zeros(self.n_bandits, dtype=np.float32)
        info = {}

        return obs, info

    def step(self, action):
        """Step the environment."""
        action = int(action)
        self.last_action = action

        # Create observation: one-hot encoding of the action taken
        obs = np.zeros(self.n_bandits, dtype=np.float32)
        obs[action] = 1.0

        # Base reward is 0 - actual reward computed by wrapper
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info
