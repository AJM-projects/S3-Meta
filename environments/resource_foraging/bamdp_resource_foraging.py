"""Resource Foraging BAMDP wrapper."""

import numpy as np
from environments.bamdpdeepmind import BamdpDeepmind


class BamdpResourceForaging(BamdpDeepmind):
    """
    Resource Foraging environment for meta-RL.

    Agent must learn which resource patches have the highest abundance
    based on task parameters. Different tasks have different patch quality.
    """

    def __init__(self, env, vae, args, tasks=None):
        self.n_patches = args.n_patches
        self.task_param_dim = args.task_param_dim
        super().__init__(env, vae, args, tasks)

    def get_initial_env_params(self):
        """Return initial placeholder parameters."""
        return np.zeros(self.task_param_dim)

    def set_parameters(self, task):
        """Set the task parameters that determine patch resource abundances."""
        self.current_env_params = np.array(task)

        # Create patch abundance from task parameters
        if self.task_param_dim >= self.n_patches:
            # Use first n_patches parameters as abundances
            self.patch_abundances = self.current_env_params[: self.n_patches]
        else:
            # Cycle through task parameters to cover all patches
            self.patch_abundances = np.array(
                [
                    self.current_env_params[i % self.task_param_dim]
                    for i in range(self.n_patches)
                ]
            )

        # Ensure abundances are positive (map to 0-2 range)
        self.patch_abundances = np.abs(self.patch_abundances)

    def process_sar(self, obs, action, reward):
        action_one_hot = np.zeros(5, dtype=np.float32)
        action_one_hot[int(action)] = 1.0
        return obs, action_one_hot, reward

    def get_reward(self, reward, obs, info):
        """
        Compute reward based on foraging success and patch abundances.
        """
        # Track previous resource levels to detect foraging
        if not hasattr(self, "prev_patch_resources"):
            self.prev_patch_resources = obs[2:]  # Skip agent position
            return 0.0

        current_patch_resources = obs[2:]  # Skip agent position

        # Calculate total reward from all patches that were foraged
        total_reward = 0.0
        for patch_idx in range(self.n_patches):
            # Check if resources decreased (indicating foraging)
            resource_consumed = max(
                0,
                self.prev_patch_resources[patch_idx]
                - current_patch_resources[patch_idx],
            )

            if resource_consumed > 0:
                # Reward = abundance * amount consumed
                patch_reward = self.patch_abundances[patch_idx] * resource_consumed
                total_reward += patch_reward

        # Update previous resources for next step
        self.prev_patch_resources = current_patch_resources.copy()

        return total_reward

    def reset(self, **kwargs):
        """Reset the environment and clear resource tracking."""
        result = super().reset(**kwargs)
        # Clear previous resource tracking
        if hasattr(self, "prev_patch_resources"):
            delattr(self, "prev_patch_resources")
        return result
