"""Minigrid BAMDP Wrappers.

This module defines a collection of wrapper classes for creating
meta–environments on top of the Farama MiniGrid suite. Each wrapper
extends the :class:`BamdpDeepmind` interface found in
``bamdpdeepmind.py``. They adapt Minigrid environments to operate
within a Bayes–adaptive meta–learning framework by augmenting
observations with belief vectors and exposing task–dependent
transitions and reward functions.

The classes defined here follow the same design pattern found in
``bamdpdeepmind.py``: every wrapper must implement
``get_initial_env_params`` and ``set_parameters``. They are written
following the PEP 8 and PEP 257 conventions and include detailed
docstrings for documentation.

"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environments.bamdpdeepmind import BamdpDeepmind
from environments.minigrid_param_envs import (
    KeyDoorTwoColorEnv,
    TwoGoalEnv,
)


class BamdpMinigridBase(BamdpDeepmind):
    """Base BAMDP wrapper for Minigrid tasks.

    MiniGrid environments often return observations as dictionaries
    containing an RGB image, the agent's direction and a mission string.
    The base :class:`BamdpDeepmind` assumes a flat numeric observation.
    This class provides helper methods to flatten the numeric portion
    of such observations and to construct augmented observation spaces
    that include belief vectors. Subclasses should override
    :meth:`get_initial_env_params` and :meth:`set_parameters` to
    implement their task distributions.
    """

    def _flatten_obs(self, obs: Any) -> np.ndarray:
        """Flatten a Minigrid observation into a one–dimensional array.

        Parameters
        ----------
        obs : Any
            The observation returned by the underlying environment.

        Returns
        -------
        np.ndarray
            A one–dimensional array containing concatenated numeric
            components of ``obs``. Mission strings or other non–numeric
            values are ignored.
        """
        if isinstance(obs, dict):
            parts: list[np.ndarray] = []
            for value in obs.values():
                if isinstance(value, np.ndarray):
                    parts.append(value.astype(np.float32).flatten())
                elif np.isscalar(value) and not isinstance(value, str):
                    parts.append(np.array([value], dtype=np.float32))
            if parts:
                return np.concatenate(parts)
            return np.array([], dtype=np.float32)
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32).flatten()
        return np.array(obs, dtype=np.float32).flatten()

    def create_new_observation_space(self) -> gym.Space:
        """Create a flattened observation space augmented with a belief vector.

        Returns
        -------
        gym.Space
            A :class:`gymnasium.spaces.Box` describing the bounds of
            flattened observations concatenated with the belief vector.
        """
        # Determine belief dimension based on oracle setting.
        belief_dim = self.task_dim if self.is_oracle else len(self.prior)
        orig_space = self.env.observation_space
        # If the original is a Box, we can propagate bounds
        if isinstance(orig_space, spaces.Box):
            try:
                original_low = orig_space.low.reshape(-1)
                original_high = orig_space.high.reshape(-1)
                extra_inf = np.full((belief_dim,), np.inf)
                extra_minus = np.full((belief_dim,), -np.inf)
                new_low = np.concatenate((original_low, extra_minus))
                new_high = np.concatenate((original_high, extra_inf))
                return spaces.Box(
                    low=new_low,
                    high=new_high,
                    shape=(len(new_low),),
                    dtype=orig_space.dtype,
                )
            except Exception:
                pass
        # For Dict or other spaces, fall back to sampling to infer shape
        try:
            sample = self.env.observation_space.sample()
        except Exception:
            # If sampling fails, try a real reset
            sample, _ = self.env.reset()
        flat = self._flatten_obs(sample)
        low = np.full((len(flat) + belief_dim,), -np.inf, dtype=np.float32)
        high = np.full((len(flat) + belief_dim,), np.inf, dtype=np.float32)
        return spaces.Box(low=low, high=high, shape=(len(low),), dtype=np.float32)

    def process_sar(
        self, obs: Any, action: Any, reward: float
    ) -> tuple[np.ndarray, Any, float]:
        """Flatten the observation before belief encoding.

        Returns
        -------
        tuple[np.ndarray, Any, float]
            A tuple of flattened observation, action and reward.
        """
        flat_obs = self._flatten_obs(obs)
        # One-hot encode discrete action to match action_embedding_size
        if isinstance(self.env.action_space, spaces.Discrete):
            a = int(action)
            one_hot = np.zeros(self.env.action_space.n, dtype=np.float32)
            if 0 <= a < self.env.action_space.n:
                one_hot[a] = 1.0
            action_vec = one_hot
        else:
            action_vec = np.array([action], dtype=np.float32).flatten()
        return flat_obs, action_vec, reward

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """Reset the environment and apply the next task.

        Observations are flattened and augmented with the prior belief.

        Returns
        -------
        tuple[np.ndarray, dict]
            A tuple containing the augmented observation and an info
            dictionary with the current task parameters.
        """
        obs, info = self.env.reset(**kwargs)
        self.previous_action = None
        self.previous_obs = None
        self.current_episode = 0
        self.tasks_completed += 1
        task = self.tasks[self.tasks_completed % len(self.tasks)]
        self.set_parameters(task)
        self.prior, self.hidden_state = self.get_prior()
        flat_obs = self._flatten_obs(obs)
        belief = self.current_env_params if self.is_oracle else self.prior
        augmented_obs = np.concatenate([flat_obs, belief.copy()])
        info["current_env_params"] = self.current_env_params
        return augmented_obs, info

    def step(self, action):
        # Step underlying env
        raw_next_obs, raw_reward, terminated, truncated, info = self.env.step(action)
        # Compute task-shaped reward
        reward = self.get_reward(raw_reward, raw_next_obs, info)
        # Enforce fixed-length episodes: ignore early terminations
        if self.current_timestep < self.max_episode_length - 1:
            self.current_timestep += 1
            terminated = False
            truncated = False
        else:
            self.current_timestep = 0
            terminated = True
            truncated = True
        # Prepare inputs for encoder (flatten obs, one-hot action if needed)
        flat_next_obs, enc_action, reward = self.process_sar(
            raw_next_obs, action, reward
        )
        # Update latent belief
        belief = self.update_encoding(flat_next_obs, enc_action, reward)
        augmented_obs = np.concatenate([flat_next_obs, belief])
        # Track and attach info
        self.previous_obs = flat_next_obs
        self.previous_action = enc_action
        info["current_env_params"] = self.current_env_params
        return augmented_obs, reward, terminated, truncated, info


class BamdpMinigridTwoGoal(BamdpMinigridBase):
    """BAMDP wrapper for `TwoGoalEnv`.

    Task vector: ``[target_goal, step_penalty, layout_id]``
      - target_goal in {0, 1}: which goal gives terminal reward 1.0
      - step_penalty >= 0: per–step penalty encouraging fast inference
      - layout_id in {0, 1, 2}: selects different wall configurations

    The wrapper replaces the intrinsic reward with:
      reward = 1.0 if reached_goal_index == target_goal else -step_penalty
    """

    def get_initial_env_params(self) -> np.ndarray:
        return np.array([0.0, 0.01, 0.0], dtype=float)

    def set_parameters(self, task: Any) -> None:
        param = np.array(task, dtype=float).flatten()
        if param.size < 3:
            param = np.concatenate([param, np.zeros(3 - param.size, dtype=float)])
        target_goal = int(param[0]) % 2
        step_penalty = float(param[1])
        layout_id = int(param[2]) % 3
        self.current_env_params = np.array(
            [target_goal, step_penalty, layout_id], dtype=float
        )

        # Ensure underlying env is a TwoGoalEnv; if not, wrap/replace it.
        if not isinstance(self.env, TwoGoalEnv):
            # Try to reconstruct with same render_mode if possible
            render_mode = getattr(self.env, "render_mode", None)
            self.env = TwoGoalEnv(render_mode=render_mode)
        # Apply task to env to change transitions
        self.env.set_task_params(target_goal, step_penalty, layout_id)
        self.observation_space = self.create_new_observation_space()

    def get_reward(self, reward: float, obs: Any, info: dict) -> float:
        target_goal = int(self.current_env_params[0])
        step_penalty = float(self.current_env_params[1])
        reached = int(info.get("reached_goal_index", -1))
        if reached == target_goal:
            return 1.0
        return -step_penalty


class BamdpMinigridKeyDoor(BamdpMinigridBase):
    """BAMDP wrapper for KeyDoorTwoColorEnv.

    Task vector: [target_color, step_penalty, layout_id]
      - target_color in {0 (red), 1 (blue)} determines which goal yields 1.0
      - step_penalty >= 0 applied each step when not on the correct goal
      - layout_id selects internal wall configuration
    """

    def get_initial_env_params(self) -> np.ndarray:
        return np.array([0.0, 0.01, 0.0], dtype=float)

    def set_parameters(self, task: Any) -> None:
        param = np.array(task, dtype=float).flatten()
        if param.size < 3:
            param = np.concatenate([param, np.zeros(3 - param.size, dtype=float)])
        target_color = int(param[0]) % 2
        step_penalty = float(param[1])
        layout_id = int(param[2]) % 3
        self.current_env_params = np.array(
            [target_color, step_penalty, layout_id], dtype=float
        )
        if not isinstance(self.env, KeyDoorTwoColorEnv):
            render_mode = getattr(self.env, "render_mode", None)
            self.env = KeyDoorTwoColorEnv(render_mode=render_mode)
        self.env.set_task_params(target_color, step_penalty, layout_id)
        self.observation_space = self.create_new_observation_space()

    def get_reward(self, reward: float, obs: Any, info: dict) -> float:
        target_color = int(self.current_env_params[0])
        step_penalty = float(self.current_env_params[1])
        reached = int(info.get("reached_goal_color_id", -1))
        if reached == target_color:
            return 1.0
        return -step_penalty


class BamdpTemporalMemoryMaze(BamdpMinigridBase):
    """
    Temporal Memory Maze environment for meta-RL testing long-range memory.

    This environment tests the agent's ability to:
    1. Observe and encode early visual cues (colored objects)
    2. Maintain this information through long navigation sequences
    3. Retrieve and apply the memory at decision points
    4. Handle visual distractors during the memory maintenance phase

    Different tasks vary in:
    - Cue-to-exit color mappings
    - Maze layouts (linear, L-shaped, spiral)
    - Distractor object placements

    This is perfect for testing SSMs vs RNNs on visual working memory
    and long-range dependencies in partially observable environments.
    """

    def get_initial_env_params(self):
        """Return initial placeholder parameters."""
        # [cue_color_1_idx, exit_color_1_idx, cue_color_2_idx, exit_color_2_idx, maze_layout]
        return np.zeros(5)

    def set_parameters(self, task):
        """Set the task parameters: cue mappings and maze layout."""
        self.current_env_params = np.array(task)

        # Extract task components
        cue1_idx, exit1_idx, cue2_idx, exit2_idx, maze_layout = task

        # Convert indices to color names
        available_colors = ["red", "blue", "green", "yellow", "purple", "grey"]
        cue1_color = available_colors[int(cue1_idx) % len(available_colors)]
        exit1_color = available_colors[int(exit1_idx) % len(available_colors)]
        cue2_color = available_colors[int(cue2_idx) % len(available_colors)]
        exit2_color = available_colors[int(exit2_idx) % len(available_colors)]

        # Create cue mapping
        cue_mapping = {cue1_color: exit1_color, cue2_color: exit2_color}

        # Set parameters in the underlying environment
        self.env.set_task_params(cue_mapping, int(maze_layout))

        # Store for oracle mode
        self.cue_mapping = cue_mapping
        self.maze_layout = int(maze_layout)

        # Update observation space
        self.observation_space = self.create_new_observation_space()

    def process_sar(self, obs, action, reward):
        """Process state-action-reward for the VAE encoder (MiniGrid version)."""
        # Flatten the MiniGrid observation
        flat_obs = self._flatten_obs(obs)

        # Create one-hot encoding for actions (MiniGrid has 7 actions)
        action_one_hot = np.zeros(7, dtype=np.float32)
        if action < 7:
            action_one_hot[int(action)] = 1.0

        return flat_obs, action_one_hot, reward

    def get_reward(self, reward, obs, info):
        """
        Enhanced reward based on phase and memory performance.
        """
        base_reward = reward

        # Phase-based reward shaping
        phase = info.get("phase", "cue")

        if phase == "cue":
            # Small reward for observing cues
            if len(info.get("cues_seen", [])) > len(
                getattr(self, "_prev_cues_seen", [])
            ):
                base_reward += 0.1
        elif phase == "navigate":
            # Small penalty to encourage efficient navigation
            base_reward -= 0.01
        elif phase == "decision":
            # Main reward comes from correct exit choice
            # This is handled by the base environment
            pass

        # Store previous state for comparison
        self._prev_cues_seen = info.get("cues_seen", [])

        return base_reward
