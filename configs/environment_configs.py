"""Environment-specific configurations."""

import numpy as np
from configs.task_generators import (
    generate_valid_point_mass_tasks_structured,
    generate_twogoal_tasks,
    generate_keydoor_tasks,
    generate_memory_tasks,
    generate_mab_tasks,
    generate_mab_n_tasks,
    generate_resource_foraging_tasks,
    generate_temporal_maze_tasks,
    generate_delayed_mab_tasks,
)

# Set seeds for reproducible task generation
np.random.seed(42)
cheetah_train_all = [np.random.uniform(0.5, 3) for _ in range(50)]
np.random.seed(123)
cheetah_test_all = [np.random.uniform(0.5, 3) for _ in range(5)]

np.random.seed(42)
swimmer_train_all = [
    (np.random.uniform(-10, 10), np.random.uniform(0, 1)) for _ in range(50)
]
np.random.seed(123)
swimmer_test_all = [
    (np.random.uniform(-10, 10), np.random.uniform(0, 1)) for _ in range(5)
]

minigrid_twogoal_train = generate_twogoal_tasks(num_tasks=60, seed=42)
minigrid_twogoal_test = generate_twogoal_tasks(num_tasks=20, seed=123)

minigrid_keydoor_train = generate_keydoor_tasks(num_tasks=60, seed=24)
minigrid_keydoor_test = generate_keydoor_tasks(num_tasks=20, seed=48)

minigrid_memory_train = generate_memory_tasks(num_tasks=60, seed=7)
minigrid_memory_test = generate_memory_tasks(num_tasks=20, seed=77)

mab_train_tasks = generate_mab_tasks(num_tasks=50, seed=42)
mab_test_tasks = generate_mab_tasks(num_tasks=20, seed=123)

mab_n_train_tasks = generate_mab_n_tasks(num_tasks=50, n_bandits=10, seed=42)
mab_n_test_tasks = generate_mab_n_tasks(num_tasks=20, n_bandits=10, seed=123)

foraging_train_tasks = generate_resource_foraging_tasks(num_tasks=50, seed=42)
foraging_test_tasks = generate_resource_foraging_tasks(num_tasks=20, seed=456)

temporal_maze_train_tasks = generate_temporal_maze_tasks(num_tasks=60, seed=42)
temporal_maze_test_tasks = generate_temporal_maze_tasks(num_tasks=20, seed=999)

delayed_mab_train_tasks = generate_delayed_mab_tasks(num_tasks=60, seed=42)
delayed_mab_test_tasks = generate_delayed_mab_tasks(num_tasks=20, seed=777)

point_mass_easy = {
    "state_dim": 4,
    "action_dim": 2,
    "policy_kwargs": None,
    "train_tasks": generate_valid_point_mass_tasks_structured(100, 0.1, 42),
    "test_tasks": generate_valid_point_mass_tasks_structured(20, 0.1, 84),
    "latent_dim": 2,
    "reward_decoder_layers": [32, 16],
    "task_decoder_layers": [32, 16],
    "state_embedding_size": 4,
    "action_embedding_size": 2,
    "max_rollouts_per_task": 1,
    "max_episode_length": 500,
    "env_name": "point_mass_easy",
    "env_register": ("point_mass", "easy"),
    "env_difficulty": "easy",
    "total_timesteps": 3000000,
    "task_dim": 2,
}

point_mass_hard = {
    "state_dim": 4,
    "action_dim": 2,
    "policy_kwargs": None,
    "train_tasks": generate_valid_point_mass_tasks_structured(100, 0.1, 42),
    "test_tasks": generate_valid_point_mass_tasks_structured(20, 0.1, 84),
    "latent_dim": 2,
    "reward_decoder_layers": [32, 32],
    "task_decoder_layers": [32, 16],
    "state_embedding_size": 4,
    "action_embedding_size": 2,
    "max_rollouts_per_task": 1,
    "max_episode_length": 500,
    "env_name": "point_mass_hard",
    "env_register": ("point_mass", "easy"),
    "env_difficulty": "hard",
    "total_timesteps": 3000000,
    "task_dim": 2,
}

HalfCheetah_easy = {
    "state_dim": 17,
    "action_dim": 6,
    "policy_kwargs": None,
    "train_tasks": cheetah_train_all,
    "test_tasks": cheetah_test_all,
    "latent_dim": 2,
    "reward_decoder_layers": [32, 16],
    "task_decoder_layers": [32, 16],
    "state_embedding_size": 17,
    "action_embedding_size": 6,
    "max_rollouts_per_task": 1,
    "max_episode_length": 500,
    "env_name": "HalfCheetahVel_easy",
    "env_register": "HalfCheetah-v5",
    "total_timesteps": 3000000,
    "task_dim": 1,
    "env_difficulty": "easy",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,
}

HalfCheetah_hard = {
    "state_dim": 17,
    "action_dim": 6,
    "policy_kwargs": None,
    "train_tasks": cheetah_train_all,
    "test_tasks": cheetah_test_all,
    "latent_dim": 2,
    "reward_decoder_layers": [32, 16],
    "task_decoder_layers": [32, 16],
    "state_embedding_size": 17,
    "action_embedding_size": 6,
    "max_rollouts_per_task": 1,
    "max_episode_length": 500,
    "env_name": "HalfCheetahVel_hard",
    "env_register": "HalfCheetah-v5",
    "total_timesteps": 3000000,
    "task_dim": 1,
    "env_difficulty": "hard",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,
}

CheetahDir_config = {
    "state_dim": 17,
    "action_dim": 6,
    "policy_kwargs": None,
    "train_tasks": [1.0, -1.0],
    "test_tasks": [1.0, -1.0],
    "latent_dim": 2,
    "reward_decoder_layers": [32, 16],
    "task_decoder_layers": [32, 16],
    "state_embedding_size": 17,
    "action_embedding_size": 6,
    "max_rollouts_per_task": 1,
    "max_episode_length": 500,
    "env_name": "CheetahDir",
    "env_register": "CheetahDir",
    "total_timesteps": 3000000,
    "task_dim": 1,
    "env_difficulty": "standard",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,
}

Swimmer_easy = {
    "state_dim": 8,
    "action_dim": 2,
    "policy_kwargs": None,
    "train_tasks": swimmer_train_all,
    "test_tasks": swimmer_test_all,
    "latent_dim": 4,
    "reward_decoder_layers": [32, 16],
    "task_decoder_layers": [32, 16],
    "state_embedding_size": 8,
    "action_embedding_size": 2,
    "max_rollouts_per_task": 1,
    "max_episode_length": 500,
    "env_name": "Swimmer_easy",
    "env_register": "Swimmer-v5",
    "total_timesteps": 3000000,
    "task_dim": 2,
    "env_difficulty": "easy",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,
}

ResourceForaging_config = {
    "state_dim": 6,
    "action_dim": 5,
    "n_patches": 4,
    "grid_size": 8,
    "task_param_dim": 4,
    "policy_kwargs": None,
    "train_tasks": foraging_train_tasks,
    "test_tasks": foraging_test_tasks,
    "latent_dim": 4,
    "reward_decoder_layers": [32, 16],
    "task_decoder_layers": [32, 16],
    "state_embedding_size": 6,
    "action_embedding_size": 5,
    "max_rollouts_per_task": 1,
    "max_episode_length": 200,
    "env_name": "ResourceForaging",
    "env_register": "ResourceForaging",
    "total_timesteps": 2000000,
    "task_dim": 4,
    "env_difficulty": "standard",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,
}

MAB_config = {
    "state_dim": 5,
    "action_dim": 5,
    "n_bandits": 5,
    "task_param_dim": 3,
    "policy_kwargs": None,
    "train_tasks": mab_train_tasks,
    "test_tasks": mab_test_tasks,
    "latent_dim": 4,
    "reward_decoder_layers": [32, 16],
    "task_decoder_layers": [32, 16],
    "state_embedding_size": 5,
    "action_embedding_size": 5,
    "max_rollouts_per_task": 1,
    "max_episode_length": 100,
    "env_name": "MAB",
    "env_register": "MAB",
    "total_timesteps": 1000000,
    "task_dim": 3,
    "env_difficulty": "standard",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,
}

MAB10_config = {
    "state_dim": 10,
    "action_dim": 10,
    "n_bandits": 10,
    "task_param_dim": 10,
    "policy_kwargs": None,
    "train_tasks": mab_n_train_tasks,
    "test_tasks": mab_n_test_tasks,
    "latent_dim": 4,
    "reward_decoder_layers": [32, 16],
    "task_decoder_layers": [32, 16],
    "state_embedding_size": 10,
    "action_embedding_size": 10,
    "max_rollouts_per_task": 1,
    "max_episode_length": 100,
    "env_name": "MAB10",
    "env_register": "MAB",
    "total_timesteps": 1000000,
    "task_dim": 10,
    "env_difficulty": "standard",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,
}

MiniGridTwoGoal_config = {
    "state_dim": 148,
    "action_dim": 7,
    "policy_kwargs": None,
    "train_tasks": minigrid_twogoal_train,
    "test_tasks": minigrid_twogoal_test,
    "latent_dim": 4,
    "reward_decoder_layers": [64, 32],
    "task_decoder_layers": [64, 32],
    "state_embedding_size": 148,
    "action_embedding_size": 7,
    "max_rollouts_per_task": 1,
    "max_episode_length": 200,
    "env_name": "MiniGridTwoGoal",
    "env_register": "MiniGridTwoGoal",
    "total_timesteps": 1000000,
    "task_dim": 3,
    "env_difficulty": "standard",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,
}

MiniGridKeyDoor_config = {
    "state_dim": 148,
    "action_dim": 7,
    "policy_kwargs": None,
    "train_tasks": minigrid_keydoor_train,
    "test_tasks": minigrid_keydoor_test,
    "latent_dim": 4,
    "reward_decoder_layers": [64, 32],
    "task_decoder_layers": [64, 32],
    "state_embedding_size": 148,
    "action_embedding_size": 7,
    "max_rollouts_per_task": 1,
    "max_episode_length": 200,
    "env_name": "MiniGridKeyDoor",
    "env_register": "MiniGridKeyDoor",
    "total_timesteps": 1000000,
    "task_dim": 3,
    "env_difficulty": "standard",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,
}

TemporalMaze_config = {
    "state_dim": 148,  # MiniGrid flattened observation (7x7x3 + direction)
    "action_dim": 7,  # MiniGrid standard actions
    "policy_kwargs": None,
    "train_tasks": temporal_maze_train_tasks,
    "test_tasks": temporal_maze_test_tasks,
    "latent_dim": 4,  # Match other MiniGrid configs
    "reward_decoder_layers": [64, 32],
    "task_decoder_layers": [64, 32],
    "state_embedding_size": 148,  # Match working MiniGrid
    "action_embedding_size": 7,  # Match working MiniGrid
    "max_rollouts_per_task": 1,
    "max_episode_length": 200,
    "env_name": "TemporalMaze",
    "env_register": "TemporalMaze",
    "total_timesteps": 2000000,  # Match other configs
    "task_dim": 5,
    "env_difficulty": "standard",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,  # Match other MiniGrid configs
}

DelayedMAB_config = {
    "state_dim": 8,  # [base_payoffs(5), is_signal, is_distractor, is_decision]
    "action_dim": 5,
    "n_bandits": 5,
    "signal1_length": 10,  # Length of first signal phase
    "distractor1_length": 35,  # Length of first distractor phase
    "intermediate_decision_length": 5,  # Length of intermediate decision phase
    "signal2_length": 10,  # Length of second signal phase
    "distractor2_length": 35,  # Length of second distractor phase
    "final_decision_length": 5,  # Length of final decision phase
    "policy_kwargs": None,
    "train_tasks": delayed_mab_train_tasks,
    "test_tasks": delayed_mab_test_tasks,
    "latent_dim": 4,
    "reward_decoder_layers": [32, 16],
    "task_decoder_layers": [32, 16],
    "state_embedding_size": 8,
    "action_embedding_size": 5,
    "reward_embedding_size": 4,
    "max_rollouts_per_task": 1,
    "max_episode_length": 100,  # 25+15+10+25+15+10=100
    "env_name": "DelayedMAB",
    "env_register": "DelayedMAB",
    "total_timesteps": 3000000,
    "task_dim": 10,  # base_payoffs(5) + multipliers(5)
    "env_difficulty": "standard",
    "input_prev_action": True,
    "input_prev_state": True,
    "vae_buffer_size": 50,
}


ENVIRONMENT_CONFIGS = {
    "point_mass_easy": point_mass_easy,
    "CheetahVel": HalfCheetah_easy,
    "CheetahDir": CheetahDir_config,
    "MAB": MAB_config,
    "MAB10": MAB10_config,
    "ResourceForaging": ResourceForaging_config,
    "MiniGridTwoGoal": MiniGridTwoGoal_config,
    "MiniGridKeyDoor": MiniGridKeyDoor_config,
    "TemporalMaze": TemporalMaze_config,
    "DelayedMAB": DelayedMAB_config,
}
