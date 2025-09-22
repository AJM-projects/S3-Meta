import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ResourceForagingEnv(gym.Env):
    """
    Resource Foraging Environment.

    Agent moves in a grid world with multiple resource patches.
    Must learn which patches have the best resources based on task parameters.

    :param grid_size: Size of the grid world
    :param n_patches: Number of resource patches
    :param max_episode_length: Maximum steps per episode
    """

    def __init__(self, grid_size=8, n_patches=4, max_episode_length=100):
        super().__init__()
        self.grid_size = grid_size
        self.n_patches = n_patches
        self.max_episode_length = max_episode_length

        # Actions: 0=up, 1=down, 2=left, 3=right, 4=forage
        self.action_space = spaces.Discrete(5)

        # Observation: [agent_x, agent_y, patch1_resources, patch2_resources, ...]
        obs_dim = 2 + n_patches  # position + resource levels
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Fixed patch locations (corners and center areas)
        self.patch_locations = self._generate_patch_locations()

        # Initialize state
        self.agent_pos = np.array([0, 0])
        self.patch_resources = np.ones(n_patches)  # Current resource levels
        self.step_count = 0

    def _generate_patch_locations(self):
        """Generate fixed locations for resource patches."""
        locations = []
        if self.n_patches >= 1:
            locations.append([1, 1])  # Top-left area
        if self.n_patches >= 2:
            locations.append([self.grid_size - 2, 1])  # Top-right area
        if self.n_patches >= 3:
            locations.append([1, self.grid_size - 2])  # Bottom-left area
        if self.n_patches >= 4:
            locations.append(
                [self.grid_size - 2, self.grid_size - 2]
            )  # Bottom-right area
        if self.n_patches >= 5:
            locations.append([self.grid_size // 2, self.grid_size // 2])  # Center

        # Add more patches randomly if needed
        while len(locations) < self.n_patches:
            x = np.random.randint(1, self.grid_size - 1)
            y = np.random.randint(1, self.grid_size - 1)
            if [x, y] not in locations:
                locations.append([x, y])

        return np.array(locations)

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset agent to random position
        self.agent_pos = np.array(
            [np.random.randint(0, self.grid_size), np.random.randint(0, self.grid_size)]
        )

        # Reset resource levels
        self.patch_resources = np.ones(self.n_patches)
        self.step_count = 0

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """Step the environment."""
        self.step_count += 1

        # Execute action
        if action < 4:  # Movement actions
            self._move_agent(action)
        elif action == 4:  # Forage action
            self._forage()

        # Resource regeneration (slow)
        self.patch_resources = np.minimum(1.0, self.patch_resources + 0.01)

        # Get observation and compute base reward (0 - actual reward computed by wrapper)
        obs = self._get_observation()
        reward = 0.0

        # Episode termination
        terminated = False
        truncated = self.step_count >= self.max_episode_length

        info = {}

        return obs, reward, terminated, truncated, info

    def _move_agent(self, action):
        """Move the agent based on action."""
        new_pos = self.agent_pos.copy()

        if action == 0:  # Up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # Down
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == 2:  # Left
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # Right
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)

        self.agent_pos = new_pos

    def _forage(self):
        """Attempt to forage at current location."""
        # Check if agent is at a patch location
        for patch_idx, patch_loc in enumerate(self.patch_locations):
            if np.allclose(self.agent_pos, patch_loc, atol=0.5):
                # Consume some resources from this patch
                consumption = min(0.2, self.patch_resources[patch_idx])
                self.patch_resources[patch_idx] -= consumption
                break

    def _get_observation(self):
        """Get current observation."""
        # Normalize agent position
        agent_obs = self.agent_pos / (self.grid_size - 1)

        # Combine with patch resource levels
        obs = np.concatenate([agent_obs, self.patch_resources])

        return obs.astype(np.float32)
