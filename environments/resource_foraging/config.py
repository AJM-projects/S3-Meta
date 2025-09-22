"""Resource Foraging environment configurations."""

from configs.task_generators import generate_resource_foraging_tasks

foraging_train_tasks = generate_resource_foraging_tasks(num_tasks=50, seed=42)
foraging_test_tasks = generate_resource_foraging_tasks(num_tasks=20, seed=456)

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