import numpy as np
import copy
import torch as th
from dm_control.utils import rewards as dmc_rewards
from environments.base.bamdp_base import BamdpBase


class BamdpDeepmind(BamdpBase):
    """
    DeepMind-specific BAMDP wrapper implementation.

    Overrides specific methods for DeepMind Control Suite environments.
    """

    def __init__(self, env, vae, args, tasks):
        super().__init__(env, vae, args, tasks)

    def update_encoding(self, next_obs, action, reward):
        if self.is_oracle:
            return self.current_env_params
        with th.no_grad():
            latent_sample, latent_mean, latent_logvar, hidden_state = self.vae.encoder(
                actions=self.unhobble(action),
                states=self.unhobble(next_obs),
                rewards=self.unhobble([reward]),
                hidden_state=self.hidden_state,
                return_prior=False,
            )

        self.hidden_state = hidden_state
        if self.condition_on_logvar:
            return np.concatenate(
                [latent_mean.squeeze().numpy(), latent_logvar.squeeze().numpy()]
            )
        return latent_mean.squeeze().numpy()

    @staticmethod
    def unhobble(x) -> th.Tensor:
        x = np.array(x, dtype=np.float32)
        return th.from_numpy(x).unsqueeze(0).unsqueeze(1)

    def get_reward(self, reward, obs, info):
        return reward

    def step(self, action):
        # Step the underlying environment.
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        # Compute the adjusted reward.
        reward = self.get_reward(reward, next_obs, info)

        # Check if we've reached the maximum timestep for the episode.
        if self.current_timestep < self.max_episode_length - 1:
            self.current_timestep += 1
        else:
            self.current_timestep = 0
            terminated = True
            truncated = True

        next_obs, action, reward = self.process_sar(next_obs, action, reward)

        belief = self.update_encoding(next_obs, action, reward)
        augmented_obs = np.concatenate([next_obs, belief])

        # Update internal state tracking.
        self.previous_obs = next_obs
        self.previous_action = action
        info["current_env_params"] = self.current_env_params

        return augmented_obs, reward, terminated, truncated, info

    def process_sar(self, obs, action, reward):
        return obs, action, reward

    def reset(self, **kwargs):
        """Reset the TASK and compute the initial belief.
        This does not reset the underlying environment with the same task. This is handled in the step function.
        """
        obs, info = self.env.reset(**kwargs)
        self.previous_action = None
        self.previous_obs = None
        self.current_episode = 0
        self.tasks_completed += 1
        task = self.tasks[self.tasks_completed % len(self.tasks)]
        self.set_parameters(task)
        self.prior, self.hidden_state = self.get_prior()

        if self.is_oracle:
            belief = self.current_env_params
        else:
            belief = self.prior

        augmented_obs = np.concatenate([obs, copy.copy(belief)])

        return augmented_obs, info

    def load_vae(self, vae):
        self.vae = vae


class BamdpCheetahRun(BamdpDeepmind):
    def __init__(self, env, vae, args, tasks=None):
        super().__init__(env, vae, args, tasks)
        self.dt = self.env.unwrapped.dt

    def get_initial_env_params(self):
        return None

    def set_parameters(self, task):
        """
        Randomize the goal location and physical parameters for the Reacher environment.
        """
        self.current_env_params = task

    def get_reward(self, reward, obs, info):
        tip_velocity = obs[8]

        # get different between the tip velocity and the goal velocity
        reward = 1 if np.abs(tip_velocity - self.current_env_params) < 0.1 else 0
        return reward

    def get_velocity_reward(self, velocity):
        forward_reward = -1.0 * float(abs(velocity - self.current_env_params))
        return forward_reward

    def step(self, action):
        # Step the underlying environment.
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        velocity = info["x_velocity"]
        ctrl_cost = info["reward_ctrl"]

        forward_reward = self.get_velocity_reward(velocity)

        # Compute the adjusted reward.
        reward = forward_reward + ctrl_cost

        # Check if we've reached the maximum timestep for the episode.
        if self.current_timestep < self.max_episode_length - 1:
            self.current_timestep += 1
        else:
            self.current_timestep = 0
            terminated = True
            truncated = True

        belief = self.update_encoding(next_obs, action, reward)
        augmented_obs = np.concatenate([next_obs, belief])

        # Update internal state tracking.
        self.previous_obs = next_obs
        self.previous_action = action
        info["current_env_params"] = self.current_env_params

        return augmented_obs, reward, terminated, truncated, info


class BamdpCheetahDir(BamdpDeepmind):
    """
    Direction variant of DeepMind Control Cheetah (run):
    - Task +1: forward (use underlying reward)
    - Task -1: backward (reward for running backwards)
    """

    def __init__(self, env, vae, args, tasks=None):
        super().__init__(env, vae, args, tasks)
        self._run_speed = 10.0

    def get_initial_env_params(self):
        # Default to forward direction
        return np.array([1.0], dtype=np.float32)

    def set_parameters(self, task):
        # Expect scalar task: +1 for forward, -1 for backward
        if isinstance(task, (list, tuple, np.ndarray)):
            val = float(task[0])
        else:
            val = float(task)
        self.current_env_params = np.array(
            [1.0 if val >= 0 else -1.0], dtype=np.float32
        )

    def get_reward(self, reward, obs, info):
        direction = self.current_env_params[0]
        if direction >= 0:
            # Forward: use underlying reward (already computed by DMC)
            return reward
        # Backward: compute reward using negative speed
        try:
            speed = float(self.env._env.physics.speed())
        except Exception:
            # Fallback: use observed x-velocity if available in obs (heuristic index)
            # If unavailable, fallback to zero extra reward
            speed = 0.0
        return float(
            dmc_rewards.tolerance(
                -speed,
                bounds=(self._run_speed, float("inf")),
                margin=self._run_speed,
                value_at_margin=0.0,
                sigmoid="linear",
            )
        )


class BamdpDelayedMAB(BamdpDeepmind):
    """
    Delayed Multi-Armed Bandit environment for meta-RL testing robust temporal task inference.

    This environment uses a 5-phase structure where optimal arm selection requires
    integrating information from two temporally separated signal phases:
    1. Signal Phase 1: Agent receives first part of task information through rewards
    2. Distractor Phase 1: No information, zero rewards regardless of actions
    3. Signal Phase 2: Agent receives second part of task information through rewards
    4. Distractor Phase 2: No information, zero rewards regardless of actions
    5. Decision Phase: Agent chooses arms, optimal choice requires combining both signals

    Unlike DelayedSignalBandit, this cannot be "cheated" by ignoring early phases
    because: (1) both signal phases are required, (2) decision phase is too short
    for exploration, and (3) distractor phases prevent temporal correlations.
    """

    def __init__(self, env, vae, args, tasks=None):
        self.n_bandits = args.n_bandits
        super().__init__(env, vae, args, tasks)

    def get_initial_env_params(self):
        """Return initial placeholder parameters."""
        # [base_payoffs..., multipliers...]
        return np.zeros(self.n_bandits * 2)

    def set_parameters(self, task):
        """Set the task parameters: base payoffs and multipliers."""
        self.current_env_params = np.array(task)

        # Extract task components
        base_payoffs = self.current_env_params[: self.n_bandits]
        multipliers = self.current_env_params[self.n_bandits : self.n_bandits * 2]

        # Set parameters in the underlying environment
        self.env.set_task_params(base_payoffs, multipliers)

        # Store for oracle mode
        self.base_payoffs = base_payoffs
        self.multipliers = multipliers

    def process_sar(self, obs, action, reward):
        """Process state-action-reward for the VAE."""
        # Create one-hot encoding for actions
        action_one_hot = np.zeros(self.n_bandits, dtype=np.float32)
        if action < self.n_bandits:
            action_one_hot[int(action)] = 1.0
        return obs, action_one_hot, reward

    def get_reward(self, reward, obs, info):
        """
        Pass through the reward from the base environment.
        The base environment handles all the 5-phase temporal logic.
        """
        return reward
