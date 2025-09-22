"""Environment builder factory for creating BAMDP environments."""

from utils.dmc_wrapper import DMCGym
import gymnasium as gym

from .delayed_signal import DelayedMABEnv
from .point_mass import BamdpPointmass
from .mab_env import MultiArmedBanditEnv, BamdpMAB
from .resource_foraging import ResourceForagingEnv, BamdpResourceForaging
from .bamdpdeepmind import BamdpCheetahRun, BamdpCheetahDir, BamdpDelayedMAB

from .mini_grid_envs import BamdpMinigridTwoGoal, BamdpMinigridKeyDoor, BamdpTemporalMemoryMaze
from .minigrid_param_envs import TwoGoalEnv, KeyDoorTwoColorEnv, TemporalMemoryMazeEnv


def build_environment(env_name, agent, config, tasks):
    def _init():
        if isinstance(env_name, tuple):
            env = DMCGym(domain=env_name[0], task=env_name[1])
            if "point" in env_name[0]:
                env = BamdpPointmass(env, agent, config, tasks)
            else:
                raise ValueError(f"Unknown environment: {env_name[0]}")
            return env
            
        elif isinstance(env_name, str):
            if env_name == "MAB":
                env = MultiArmedBanditEnv(n_bandits=config.n_bandits)
                env = BamdpMAB(env, agent, config, tasks)
                return env
                
            elif env_name == "MAB10":
                env = MultiArmedBanditEnv(n_bandits=config.n_bandits)
                env = BamdpMAB(env, agent, config, tasks)
                return env

            elif env_name == "DelayedMAB":
                env = DelayedMABEnv(
                    n_bandits=config.n_bandits,
                    max_episode_length=config.max_episode_length,
                    signal1_length=config.signal1_length,
                    distractor1_length=config.distractor1_length,
                    intermediate_decision_length=config.intermediate_decision_length,
                    signal2_length=config.signal2_length,
                    distractor2_length=config.distractor2_length,
                    final_decision_length=config.final_decision_length
                )
                env = BamdpDelayedMAB(env, agent, config, tasks)
                return env
                
            elif env_name == "ResourceForaging":
                env = ResourceForagingEnv(
                    grid_size=config.grid_size,
                    n_patches=config.n_patches,
                    task_param_dim=config.task_param_dim,
                )
                env = BamdpResourceForaging(env, agent, config, tasks)
                return env

                
            elif env_name == "CheetahDir":
                # Build DeepMind Control cheetah run base env and wrap with direction-based BAMDP
                env = DMCGym(domain="cheetah", task="run")
                env = BamdpCheetahDir(env, agent, config, tasks)
                return env

            elif env_name == "MiniGridTwoGoal":
                env = TwoGoalEnv()
                env = BamdpMinigridTwoGoal(env, agent, config, tasks)
                return env

            elif env_name == "TemporalMaze":
                env = TemporalMemoryMazeEnv(
                    size=15,
                    max_steps=config.max_episode_length
                )
                env = BamdpTemporalMemoryMaze(env, agent, config, tasks)
                return env
                
            elif env_name == "MiniGridKeyDoor":
                env = KeyDoorTwoColorEnv()
                env = BamdpMinigridKeyDoor(env, agent, config, tasks)
                return env

            elif env_name == "CheetahVel":
                env = gym.make("HalfCheetah-v5")
                env = BamdpCheetahRun(env, agent, config, tasks)
                return env

            else:
                raise ValueError(f"Unknown environment: {env_name}")
        else:
            raise ValueError(f"Invalid environment name format: {env_name}")

    return _init