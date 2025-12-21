"""Task generation utilities for various environments."""

import numpy as np


def generate_valid_point_mass_tasks_structured(num_tasks=10, radius=0.2, seed=None):
    """Generate structured point mass tasks in a circle."""
    if seed is not None:
        np.random.seed(seed)

    angles = np.random.uniform(0, 2 * np.pi, num_tasks)
    tasks = [
        (round(radius * np.cos(angle), 3), round(radius * np.sin(angle), 3))
        for angle in angles
    ]
    return tasks


def generate_twogoal_tasks(num_tasks=50, step_penalty_range=(0.005, 0.05), seed=42):
    """Generate tasks for MiniGrid TwoGoal environment."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(num_tasks):
        target_goal = i % 2
        step_penalty = float(rng.uniform(*step_penalty_range))
        layout_id = int(rng.integers(0, 3))
        tasks.append([float(target_goal), step_penalty, float(layout_id)])
    return tasks


def generate_keydoor_tasks(num_tasks=50, step_penalty_range=(0.005, 0.05), seed=24):
    """Generate tasks for MiniGrid KeyDoor environment."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(num_tasks):
        target_color = i % 2
        step_penalty = float(rng.uniform(*step_penalty_range))
        layout_id = int(rng.integers(0, 3))
        tasks.append([float(target_color), step_penalty, float(layout_id)])
    return tasks


def generate_memory_tasks(num_tasks=50, seed=7):
    """Generate tasks for MemoryEnv."""
    rng = np.random.default_rng(seed)
    tasks = []
    for i in range(num_tasks):
        initial_object = int(i % 2)
        goal_placement = int(rng.integers(0, 2))
        task_mirror = bool(rng.integers(0, 2))
        tasks.append([float(initial_object), float(goal_placement), float(task_mirror)])
    return tasks


def generate_mab_tasks(num_tasks=20, n_bandits=5, task_param_dim=3, seed=42):
    """Generate diverse tasks for Multi-Armed Bandit environment."""
    np.random.seed(seed)
    tasks = []

    for _ in range(num_tasks):
        task_params = np.random.randn(task_param_dim) * 2.0
        tasks.append(task_params.tolist())

    return tasks


def generate_mab_n_tasks(num_tasks=20, n_bandits=5, seed=42):
    """Generate diverse tasks for Multi-Armed Bandit environment with n bandits."""
    np.random.seed(seed)
    tasks = []

    for _ in range(num_tasks):
        task_params = np.random.randn(n_bandits) * 2.0
        tasks.append(task_params.tolist())

    return tasks


def generate_resource_foraging_tasks(
    num_tasks=30, n_patches=4, task_param_dim=4, seed=42
):
    """Generate diverse tasks for Resource Foraging environment."""
    np.random.seed(seed)
    tasks = []

    for _ in range(num_tasks):
        task_params = np.random.uniform(0.5, 3.0, task_param_dim)

        if len(tasks) < 5:
            dominant_idx = np.random.randint(task_param_dim)
            task_params = np.random.uniform(0.1, 0.5, task_param_dim)
            task_params[dominant_idx] = np.random.uniform(2.0, 3.0)

        tasks.append(task_params.tolist())

    return tasks


def generate_temporal_maze_tasks(num_tasks=40, seed=42):
    """
    Generate diverse tasks for Temporal Memory Maze environment.

    Each task consists of:
    - cue1_idx, exit1_idx: First cue color -> correct exit color mapping
    - cue2_idx, exit2_idx: Second cue color -> correct exit color mapping
    - maze_layout: Which maze layout to use (0, 1, or 2)

    This creates visual working memory challenges where:
    - Different color combinations indicate different correct exits
    - Variable maze layouts test spatial memory + navigation
    - Long paths between cues and decision points test memory retention
    - Visual distractors challenge memory robustness

    :param num_tasks: Number of different tasks to generate
    :param seed: Random seed for reproducibility
    :return: List of task parameter vectors [cue1_idx, exit1_idx, cue2_idx, exit2_idx, maze_layout]
    """
    np.random.seed(seed)
    tasks = []

    for i in range(num_tasks):
        # Create diverse cue-exit mappings
        # Ensure cue colors are different from each other
        cue1_idx = i % 4  # Cycle through first 4 colors for primary cues
        cue2_idx = (i + 2) % 4  # Offset to ensure different cue colors

        # Map cues to exits with some systematic patterns
        if i % 3 == 0:
            # Pattern 1: Direct mapping (red->red, blue->blue)
            exit1_idx = cue1_idx
            exit2_idx = cue2_idx
        elif i % 3 == 1:
            # Pattern 2: Shifted mapping (red->green, blue->yellow)
            exit1_idx = (cue1_idx + 2) % 6
            exit2_idx = (cue2_idx + 2) % 6
        else:
            # Pattern 3: Inverse mapping
            exit1_idx = (5 - cue1_idx) % 6
            exit2_idx = (5 - cue2_idx) % 6

        # Random maze layout
        maze_layout = i % 3

        # Create task vector
        task_params = [
            float(cue1_idx),
            float(exit1_idx),
            float(cue2_idx),
            float(exit2_idx),
            float(maze_layout),
        ]
        tasks.append(task_params)

    return tasks


def generate_delayed_mab_tasks(num_tasks=40, n_bandits=5, seed=42):
    """
    Generate diverse tasks for Delayed Multi-Armed Bandit environment.

    Each task consists of:
    - base_payoffs: Base payoff values for each arm (signal 1)
    - multipliers: Multiplier values for each arm (signal 2)

    Final payoffs = base_payoffs * multipliers
    Agent must remember both signal 1 and signal 2 to maximize final rewards

    :param num_tasks: Number of different tasks to generate
    :param n_bandits: Number of bandit arms
    :param seed: Random seed for reproducibility
    :return: List of task parameter vectors [base_payoffs..., multipliers...]
    """
    np.random.seed(seed)
    tasks = []

    for i in range(num_tasks):
        # Generate diverse base payoffs (signal 1)
        base_payoffs = np.random.uniform(1.0, 5.0, n_bandits)

        # Generate multipliers (signal 2) - mix of positive and negative
        multipliers = np.random.choice(
            [-2, -1, 1, 2], n_bandits, p=[0.2, 0.3, 0.3, 0.2]
        )

        # Ensure there's variety in which arm is optimal at different stages
        if i % 3 == 0:
            # Sometimes make highest base payoff have negative multiplier
            max_base_idx = np.argmax(base_payoffs)
            multipliers[max_base_idx] = np.random.choice([-2, -1])
        elif i % 3 == 1:
            # Sometimes make lowest base payoff have highest positive multiplier
            min_base_idx = np.argmin(base_payoffs)
            multipliers[min_base_idx] = 2

        # Create task parameter vector: [base_payoffs..., multipliers...]
        task_params = base_payoffs.tolist() + multipliers.tolist()
        tasks.append(task_params)

    return tasks
