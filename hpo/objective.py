"""Optuna objective and pruning callback for HPO."""

import gc

import numpy as np
import optuna
from stable_baselines3.common.callbacks import CallbackList

from agents import build_agent
from agents.agent_configs import get_config
from callbacks.agent_callbacks import AgentCallback, EvaluationCallback
from environments.environment_builder import build_environment
from hpo.search_spaces import AGENT_SEARCH_SPACES
from policy_network import PPOWithInfo


class PrunableEvaluationCallback(EvaluationCallback):
    """Wraps EvaluationCallback to report intermediate rewards to an Optuna trial."""

    def __init__(self, trial: optuna.Trial, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial
        self._eval_count = 0

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._eval_count += 1
            self.trial.report(self.last_mean_reward, self._eval_count)
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return continue_training


def make_objective(
    agent_name: str,
    env_name: str,
    timesteps_fraction: float = 0.3,
    eval_freq: int = 10_000,
    log_dir: str = "hpo_logs",
):
    """
    Return an Optuna objective function for the given agent/environment pair.

    Args:
        agent_name: One of "S3_Meta", "belief", "humplik".
        env_name: Key from ENVIRONMENT_CONFIGS (e.g. "CheetahDir", "MAB").
        timesteps_fraction: Fraction of full total_timesteps to use per trial.
        eval_freq: How often (in PPO steps) to evaluate and report to Optuna.
        log_dir: Root directory for per-trial tensorboard logs.

    Returns:
        Callable[[optuna.Trial], float] — the objective.
    """
    if agent_name not in AGENT_SEARCH_SPACES:
        raise ValueError(f"No search space defined for agent '{agent_name}'")

    def objective(trial: optuna.Trial) -> float:
        config_key = (
            env_name if isinstance(env_name, str) else f"{env_name[0]}_{env_name[1]}"
        )
        config = get_config(agent_name, config_key)

        for key, value in AGENT_SEARCH_SPACES[agent_name](trial).items():
            setattr(config, key, value)

        config.total_timesteps = max(
            int(config.total_timesteps * timesteps_fraction),
            config.update_every_n,  # at least one VAE update
        )

        agent = build_agent(agent_name, config)
        train_env = build_environment(
            config.env_register, agent, config, config.train_tasks
        )()
        test_env = build_environment(
            config.env_register, agent, config, config.test_tasks
        )()

        model = PPOWithInfo(
            "MlpPolicy",
            train_env,
            verbose=0,
            tensorboard_log=f"{log_dir}/{config.env_name}/{agent_name}/trial_{trial.number}",
            n_steps=config.update_every_n,
        )

        agent_cb = AgentCallback(agent, config)
        eval_cb = PrunableEvaluationCallback(
            trial=trial,
            eval_env=test_env,
            envs_name="test_envs",
            eval_freq=eval_freq,
            verbose=0,
        )

        try:
            model.learn(
                total_timesteps=config.total_timesteps,
                callback=CallbackList([agent_cb, eval_cb]),
            )
        except optuna.TrialPruned:
            raise
        finally:
            train_env.close()
            test_env.close()
            gc.collect()

        # last_mean_reward = reward at the final evaluation (what the user cares about);
        # fall back to best if training ended before any eval completed.
        reward = eval_cb.last_mean_reward
        if reward == -np.inf:
            reward = eval_cb.best_mean_reward
        return float(reward)

    return objective
