"""Point mass BAMDP environment wrappers."""

import numpy as np
import copy
from environments.bamdpdeepmind import BamdpDeepmind


class BamdpPointmass(BamdpDeepmind):
    """
    Point mass environment wrapper for meta-RL.
    """

    def __init__(self, env, vae, args, tasks=None):
        super().__init__(env, vae, args, tasks)

    def get_initial_env_params(self):
        return copy.deepcopy(self.env._env.physics.named.model.geom_friction[:])

    def set_parameters(self, task):
        """
        Set the goal location for the Point Mass environment.
        """
        self.current_env_params = task

    def get_reward(self, reward, obs, info):
        """
        Computes a reward inversely proportional to the distance from the goal,
        but only if the agent is within 0.1 units of the goal.
        """

        position = obs[:2]
        distance = np.linalg.norm(position - self.current_env_params)

        if distance < 0.1:
            reward = 1 - (distance / 0.1)
        else:
            reward = 0

        return reward

    def reset(self, **kwargs):
        """
        Reset the environment and ensure the agent always starts at (0, 0).
        """
        _, info = super().reset(**kwargs)

        physics = self.env._env.physics
        physics.named.data.geom_xpos["pointmass"][:2] = 0.0
        physics.named.data.qpos["root_x"] = 0.0
        physics.named.data.qpos["root_y"] = 0.0
        physics.named.data.qvel["root_x"] = 0.0
        physics.named.data.qvel["root_y"] = 0.0

        physics.forward()
        obs = np.zeros(4)

        if self.is_oracle:
            belief = self.current_env_params
        else:
            belief = self.prior
        augmented_obs = np.concatenate([obs, copy.copy(belief)])

        info["current_env_params"] = self.current_env_params

        return augmented_obs, info


class BamdpPointMassHard(BamdpPointmass):
    def get_reward(self, reward, obs, info):
        """
        Computes a reward inversely proportional to the distance from the goal,
        but only if the agent is within 0.05 units of the goal.
        """
        position = obs[:2]
        distance = np.linalg.norm(position - self.current_env_params)

        if distance < 0.05:
            reward = 1 - (distance / 0.05)
        else:
            reward = 0

        return reward
