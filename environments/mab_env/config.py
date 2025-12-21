"""Multi-Armed Bandit environment configurations."""

from configs.task_generators import generate_mab_tasks, generate_mab_n_tasks

mab_train_tasks = generate_mab_tasks(num_tasks=50, seed=42)
mab_test_tasks = generate_mab_tasks(num_tasks=20, seed=123)

mab_n_train_tasks = generate_mab_n_tasks(num_tasks=50, n_bandits=10, seed=42)
mab_n_test_tasks = generate_mab_n_tasks(num_tasks=20, n_bandits=10, seed=123)

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
