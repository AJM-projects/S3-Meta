"""Multi-Armed Bandit BAMDP wrapper."""

import numpy as np
from environments.bamdpdeepmind import BamdpDeepmind


class BamdpMAB(BamdpDeepmind):
    """
    Multi-Armed Bandit environment for meta-RL.

    The environment consists of n bandits, where each bandit's reward is parameterized
    by task parameters.
    """

    def __init__(self, env, vae, args, tasks=None):
        self.n_bandits = args.n_bandits
        self.task_param_dim = args.task_param_dim
        super().__init__(env, vae, args, tasks)

    def get_initial_env_params(self):
        """Return initial placeholder parameters."""
        return np.zeros(self.task_param_dim)

    def set_parameters(self, task):
        """Set the task parameters that determine bandit reward distributions."""
        self.current_env_params = np.array(task)

        # Create bandit reward distributions from task parameters
        if self.task_param_dim >= self.n_bandits:
            # Use first n_bandits parameters as means
            self.bandit_means = self.current_env_params[: self.n_bandits]
        else:
            # Cycle through task parameters to cover all bandits
            self.bandit_means = np.array(
                [
                    self.current_env_params[i % self.task_param_dim]
                    for i in range(self.n_bandits)
                ]
            )

        # Fixed standard deviation for all bandits
        self.bandit_std = 0.3

    def get_reward(self, reward, obs, info):
        """
        Compute reward by sampling from bandit-specific distributions.
        """
        # Get the action from the observation (one-hot encoding)
        action = np.argmax(obs) if np.any(obs) else -1

        if action == -1:
            return 0.0  # No reward for initial state

        # Sample from the pre-computed distribution for the chosen bandit
        reward = np.random.normal(self.bandit_means[action], self.bandit_std)

        return reward

    def process_sar(self, obs, action, reward):
        action_one_hot = np.zeros(self.n_bandits, dtype=np.float32)
        action_one_hot[int(action)] = 1.0
        return obs, action_one_hot, reward