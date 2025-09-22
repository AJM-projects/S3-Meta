"""Base BAMDP environment wrapper class."""

import gymnasium as gym
import torch
from gymnasium import spaces
import numpy as np
import abc
import torch as th


class BamdpBase(gym.Wrapper):
    """
    Base class for BAMDP environment wrappers.
    
    Wraps environments to augment observations with VAE beliefs,
    handles both sample generation and VAE.
    """

    def __init__(self, env, vae, args, tasks):
        super().__init__(env)
        self.args = args
        self.vae = vae
        self.is_oracle = args.is_oracle
        self.condition_on_logvar = (
            args.condition_on_logvar if hasattr(args, "condition_on_logvar") else True
        )
        tasks = self.get_task_dim(tasks)
        self.tasks = tasks
        self.last_obs = None
        self.prior, self.hidden_state = self.get_prior()
        self.observation_space = self.create_new_observation_space()
        assert args.max_rollouts_per_task > 0, (
            "Number of episodes must be greater than 0."
        )
        self.max_episodes = args.max_rollouts_per_task
        self.max_episode_length = args.max_episode_length
        self.current_episode = 0
        self.tasks_completed = 0
        self.current_timestep = 0
        self.initial_env_params = self.get_initial_env_params()
        self.current_env_params = None

    @torch.no_grad()
    def get_prior(self):
        latent_sample, latent_mean, latent_logvar, hidden_state = (
            self.vae.encoder.prior(batch_size=1, sample=False)
        )
        if self.condition_on_logvar:
            prior_belief = np.concatenate(
                [latent_mean.squeeze().numpy(), latent_logvar.squeeze().numpy()]
            )
        else:
            prior_belief = latent_mean.squeeze().numpy()
        prior_hidden_state = hidden_state

        return prior_belief, prior_hidden_state

    def get_task_dim(self, tasks):
        if isinstance(tasks[0], (int, float)):
            self.task_dim = 1
            tasks = [[task] for task in tasks]
        else:
            self.task_dim = len(tasks[0])
        return tasks

    @abc.abstractmethod
    def get_initial_env_params(self):
        raise NotImplementedError

    @abc.abstractmethod
    def set_parameters(self, task):
        """Sample and set new parameters."""
        raise NotImplementedError

    def get_initial_hidden_state(self):
        return th.zeros(self.args.num_gru_layers, self.vae.encoder.hidden_size)

    def create_new_observation_space(self):
        if self.is_oracle:
            belief_dim = self.task_dim
        else:
            belief_dim = len(self.prior)
        original_low = self.env.observation_space.low
        original_high = self.env.observation_space.high

        extra_inf = np.full((belief_dim,), np.inf)
        extra_minus = np.full((belief_dim,), -np.inf)

        new_low = np.concatenate((original_low, extra_minus))
        new_high = np.concatenate((original_high, extra_inf))

        return spaces.Box(
            low=new_low,
            high=new_high,
            shape=(self.env.observation_space.shape[0] + belief_dim,),
            dtype=self.env.observation_space.dtype,
        )

    def update_encoding(self, next_obs, action, reward):
        if self.is_oracle:
            return np.array(self.current_env_params)

        next_obs = th.tensor(next_obs, dtype=th.float32).unsqueeze(0).unsqueeze(0)
        action = th.tensor(action, dtype=th.float32).unsqueeze(0).unsqueeze(0)
        reward = th.tensor(reward, dtype=th.float32).unsqueeze(0).unsqueeze(0)

        latent_sample, latent_mean, latent_logvar, new_hidden = (
            self.vae.encoder.forward(
                next_obs, action, reward, self.hidden_state, sample=False
            )
        )

        self.hidden_state = new_hidden.detach()

        if self.condition_on_logvar:
            next_belief = np.concatenate(
                [latent_mean.squeeze().numpy(), latent_logvar.squeeze().numpy()]
            )
        else:
            next_belief = latent_mean.squeeze().numpy()

        return next_belief

    def augment_observation(self, obs, belief_augment=None):
        if belief_augment is None:
            if self.is_oracle:
                belief_augment = np.array(self.current_env_params)
            else:
                belief_augment = self.prior

        augmented_obs = np.concatenate((obs, belief_augment))
        self.last_obs = augmented_obs
        return augmented_obs

    def sample_task_from_rollout(self):
        task_idx = np.random.randint(0, len(self.tasks))
        sampled_task = self.tasks[task_idx]
        return sampled_task

    def reset(self, **kwargs):
        if self.current_episode == 0 or self.current_episode % self.max_episodes == 0:
            task = self.sample_task_from_rollout()
            self.set_parameters(task)
            self.current_env_params = task
            self.tasks_completed += 1

        obs, info = self.env.reset(**kwargs)
        self.current_episode += 1
        self.current_timestep = 0

        if self.current_episode > 1:
            self.prior, self.hidden_state = self.get_prior()

        augmented_obs = self.augment_observation(obs)
        return augmented_obs, info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_timestep += 1

        if self.current_timestep >= self.max_episode_length:
            truncated = True

        if self.current_episode > 1:
            belief = self.update_encoding(next_obs, action, reward)
            augmented_obs = self.augment_observation(next_obs, belief)
        else:
            augmented_obs = self.augment_observation(next_obs)

        return augmented_obs, reward, terminated, truncated, info