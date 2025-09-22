"""Point mass environment configurations."""

from configs.task_generators import generate_valid_point_mass_tasks_structured

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
