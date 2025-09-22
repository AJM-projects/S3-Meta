from __future__ import annotations

from typing import Any, Tuple
import random
import numpy as np
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal, Wall
from minigrid.core.mission import MissionSpace
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.world_object import Door, Key
from minigrid.core.actions import Actions
from minigrid.core.world_object import Ball, Floor


class TwoGoalEnv(MiniGridEnv):
    """A simple parameterized MiniGrid environment with two goal tiles.

    Task parameters control which goal is rewarding, the per–step penalty,
    and a discrete layout that changes the transition dynamics (walls).

    Task vector semantics: [target_goal, step_penalty, layout_id]
      - target_goal: int in {0, 1}
      - step_penalty: float >= 0
      - layout_id: int in {0, 1, 2} (selects among predefined wall layouts)

    The environment emits info['reached_goal_index'] in step() indicating
    which goal (0/1) the agent is currently on, or -1 if on neither.
    """

    def __init__(
        self,
        size: int = 9,
        agent_start_pos: Tuple[int, int] | None = None,
        agent_start_dir: int = 0,
        default_task: Tuple[int, float, int] = (0, 0.01, 0),
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._agent_start_pos = agent_start_pos
        self._agent_start_dir = agent_start_dir

        if max_steps is None:
            max_steps = 4 * size**2

        # Runtime task parameters
        self.target_goal: int = int(default_task[0])
        self.step_penalty: float = float(default_task[1])
        self.layout_id: int = int(default_task[2])

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

        self.mission = "Find the correct goal."

    # -------------------- Task configuration --------------------
    def set_task_params(
        self, target_goal: int, step_penalty: float, layout_id: int
    ) -> None:
        self.target_goal = int(target_goal) % 2
        self.step_penalty = float(step_penalty)
        self.layout_id = int(layout_id) % 3

    # -------------------- MiniGrid overrides --------------------
    @staticmethod
    def _gen_mission() -> str:
        return "Find the correct goal."

    def _goal_positions(self, width: int, height: int) -> list[tuple[int, int]]:
        # Fixed, symmetric positions to simplify credit assignment
        return [(1, height - 2), (width - 2, height - 2)]

    def _gen_grid(self, width: int, height: int) -> None:
        # Create an empty grid and outer walls
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # Add internal walls depending on layout_id to vary transitions
        if self.layout_id == 0:
            # Vertical divider with a central gap
            for y in range(1, height - 1):
                if y == height // 2:
                    continue
                self.grid.set(width // 2, y, Wall())
        elif self.layout_id == 1:
            # Horizontal divider with a left gap
            for x in range(1, width - 1):
                if x == 1:
                    continue
                self.grid.set(x, height // 2, Wall())
        else:
            # Two small pillars creating mild detours
            self.grid.set(width // 3, height // 3, Wall())
            self.grid.set(2 * width // 3, 2 * height // 3, Wall())

        # Place two goal tiles at fixed symmetric positions
        goal_positions = self._goal_positions(width, height)
        for gx, gy in goal_positions:
            self.put_obj(Goal(), gx, gy)

        # Agent placement
        if self._agent_start_pos is not None:
            self.agent_pos = self._agent_start_pos
            self.agent_dir = self._agent_start_dir
        else:
            # Spawn near bottom center
            cx = width // 2
            self.place_agent(top=(cx - 1, 1), size=(3, 3))

        # Static mission text (kept generic to avoid leaking task params)
        self.mission = "Find the correct goal."

    def step(self, action: int):  # type: ignore[override]
        obs, reward, terminated, truncated, info = super().step(action)

        # Determine if the agent stands on one of the goal squares
        reached_idx = -1
        cell = self.grid.get(*self.agent_pos)
        if isinstance(cell, Goal):
            # Identify which goal based on position
            goal_positions = self._goal_positions(self.width, self.height)
            try:
                reached_idx = goal_positions.index(tuple(self.agent_pos))
            except ValueError:
                reached_idx = -1
        info = dict(info)
        info["reached_goal_index"] = reached_idx

        # Keep the base reward unchanged; BAMDP wrapper will compute final reward
        return obs, reward, terminated, truncated, info


class KeyDoorTwoColorEnv(MiniGridEnv):
    """Parameterized MiniGrid with two colors: keys, locked doors, colored goals.

    Task vector: [target_color, step_penalty, layout_id]
      - target_color: int in {0 (red), 1 (blue)}. Only that color's goal yields success.
      - step_penalty: non-negative float applied each step via wrapper.
      - layout_id: int in {0,1,2} selecting different internal wall patterns.

    Emits info['reached_goal_color_id'] in step: 0,1 when on that colored goal, else -1.
    """

    def __init__(
        self,
        size: int = 9,
        agent_start_pos: Tuple[int, int] | None = None,
        agent_start_dir: int = 0,
        default_task: Tuple[int, float, int] = (0, 0.01, 0),
        max_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._agent_start_pos = agent_start_pos
        self._agent_start_dir = agent_start_dir

        if max_steps is None:
            max_steps = 4 * size**2

        self.target_color: int = int(default_task[0]) % 2
        self.step_penalty: float = float(default_task[1])
        self.layout_id: int = int(default_task[2]) % 3

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )
        self.mission = "Unlock the correct colored door and reach the goal."

    @staticmethod
    def _gen_mission() -> str:
        return "Unlock the correct colored door and reach the goal."

    def set_task_params(
        self, target_color: int, step_penalty: float, layout_id: int
    ) -> None:
        self.target_color = int(target_color) % 2
        self.step_penalty = float(step_penalty)
        self.layout_id = int(layout_id) % 3

    def _place_layout_walls(self, width: int, height: int) -> None:
        if self.layout_id == 0:
            # Simple divider with gap
            for y in range(1, height - 1):
                if y == height // 2:
                    continue
                self.grid.set(width // 2, y, Wall())
        elif self.layout_id == 1:
            # Offset divider with two gaps
            for y in range(1, height - 1):
                if y in (2, height - 3):
                    continue
                self.grid.set(width // 2, y, Wall())
        else:
            # Pillars
            self.grid.set(width // 3, height // 3, Wall())
            self.grid.set(2 * width // 3, 2 * height // 3, Wall())

    def _gen_grid(self, width: int, height: int) -> None:
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self._place_layout_walls(width, height)

        # Colors: 0 red, 1 blue
        colors = ("red", "blue")
        # Left room: red set; Right room: blue set
        red_key_pos = (2, height - 3)
        red_door_pos = (width // 2, height - 4)
        red_goal_pos = (1, 1)

        blue_key_pos = (width - 3, height - 3)
        blue_door_pos = (width // 2, 3)
        blue_goal_pos = (width - 2, 1)

        # Doors locked
        self.grid.set(red_door_pos[0], red_door_pos[1], Door(colors[0], is_locked=True))
        self.grid.set(
            blue_door_pos[0], blue_door_pos[1], Door(colors[1], is_locked=True)
        )
        # Keys
        self.grid.set(red_key_pos[0], red_key_pos[1], Key(colors[0]))
        self.grid.set(blue_key_pos[0], blue_key_pos[1], Key(colors[1]))
        # Goals behind doors
        self.put_obj(Goal(), *red_goal_pos)
        self.put_obj(Goal(), *blue_goal_pos)

        # Agent
        if self._agent_start_pos is not None:
            self.agent_pos = self._agent_start_pos
            self.agent_dir = self._agent_start_dir
        else:
            self.place_agent(top=(1, height - 3), size=(3, 3))

        self.mission = "Unlock the correct colored door and reach the goal."

    def step(self, action: int):  # type: ignore[override]
        obs, reward, terminated, truncated, info = super().step(action)
        # Detect whether agent is standing on a colored goal tile (left/right side)
        reached_color = -1
        cell = self.grid.get(*self.agent_pos)
        if isinstance(cell, Goal):
            # Determine which side of the map: left => red(0), right => blue(1)
            reached_color = 0 if self.agent_pos[0] < self.width // 2 else 1
        info = dict(info)
        info["reached_goal_color_id"] = reached_color
        return obs, reward, terminated, truncated, info



class TemporalMemoryMazeEnv(MiniGridEnv):
    """
    TemporalMemoryMaze Environment for testing long-range temporal dependencies.

    The agent must:
    1. OBSERVE early cues (colored objects) that indicate the correct exit
    2. NAVIGATE through a complex maze with distractor objects
    3. REMEMBER the cues when reaching the final decision point
    4. CHOOSE the correct colored exit based on early memory

    This tests the ability to maintain task-relevant information across
    long sequences with intervening distractors - perfect for comparing
    RNNs vs SSMs on vanishing gradient problems.
    """

    def __init__(
            self,
            size=15,
            cue_mapping=None,  # Dict mapping cue colors to exit colors
            maze_layout=0,  # Which maze variant to use
            max_steps=200,
            **kwargs
    ):

        self.size = size
        self.cue_mapping = cue_mapping or {"red": "green", "blue": "yellow"}
        self.maze_layout = maze_layout
        self.phase = "cue"  # "cue", "navigate", "decision"
        self.cues_seen = []
        self.correct_exit_color = None
        self.step_count = 0

        # Available colors for cues and exits
        self.cue_colors = list(self.cue_mapping.keys())
        self.exit_colors = list(self.cue_mapping.values()) + ["purple", "grey"]  # Add distractors

        mission_space = MissionSpace(mission_func=self._gen_mission)
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            **kwargs
        )

    @staticmethod
    def _gen_mission():
        return "observe the cue, navigate through the maze, and choose the correct colored exit"

    def _gen_grid(self, width, height):
        # Create grid
        self.grid = Grid(self.width, self.height)

        # Fill borders with walls
        self.grid.wall_rect(0, 0, self.width, self.height)

        # Generate maze based on layout
        if self.maze_layout == 0:
            self._gen_maze_layout_0()
        elif self.maze_layout == 1:
            self._gen_maze_layout_1()
        else:
            self._gen_maze_layout_2()

        # Place agent at start
        self.agent_pos = (1, 1)
        self.agent_dir = 0

        self.mission = self._gen_mission()

    def _gen_maze_layout_0(self):
        """Generate maze layout 0: Linear path with cue room and exit junction"""

        # Cue room (top-left area)
        self._create_cue_room(2, 2, 4, 3)

        # Corridor connecting cue room to navigation area
        for x in range(6, 10):
            self.grid.set(x, 3, None)

        # Navigation corridor with distractors
        for y in range(4, 12):
            self.grid.set(9, y, None)
            # Add distractor objects
            if y % 3 == 0 and y > 4:
                color = random.choice(["purple", "grey", "red", "blue"])
                obj = Ball(color) if random.random() > 0.5 else Key(color)
                self.grid.set(8, y, obj)

        # Decision junction at bottom
        self._create_decision_area(7, 12, 5, 2)

    def _gen_maze_layout_1(self):
        """Generate maze layout 1: L-shaped path"""

        # Cue room (top-right)
        self._create_cue_room(10, 2, 4, 3)

        # Horizontal corridor
        for x in range(2, 10):
            self.grid.set(x, 3, None)

        # Vertical corridor with distractors
        for y in range(4, 11):
            self.grid.set(2, y, None)
            # Add distractors
            if y % 2 == 1:
                color = random.choice(["purple", "grey"] + self.cue_colors)
                self.grid.set(3, y, Key(color))

        # Decision area
        self._create_decision_area(1, 11, 6, 2)

    def _gen_maze_layout_2(self):
        """Generate maze layout 2: Spiral path"""

        # Cue room (center-left)
        self._create_cue_room(2, 6, 3, 3)

        # Spiral corridor
        path_coords = [
            # Go right
            *[(x, 7) for x in range(5, 11)],
            # Go up
            *[(10, y) for y in range(6, 3, -1)],
            # Go left
            *[(x, 3) for x in range(9, 6, -1)],
            # Go down
            *[(7, y) for y in range(4, 11)],
            # Go right to exit
            *[(x, 11) for x in range(8, 12)]
        ]

        for i, (x, y) in enumerate(path_coords):
            self.grid.set(x, y, None)
            # Add distractors periodically
            if i % 5 == 4 and i > 0:
                distractor_x, distractor_y = path_coords[i - 1]
                if self.grid.get(distractor_x + 1, distractor_y) is None:
                    color = random.choice(["purple", "grey"] + self.cue_colors)
                    self.grid.set(distractor_x + 1, distractor_y, Ball(color))

        # Decision area
        self._create_decision_area(11, 10, 3, 3)

    def _create_cue_room(self, x, y, width, height):
        """Create a room with cue objects"""
        # Clear the room area
        for dx in range(width):
            for dy in range(height):
                self.grid.set(x + dx, y + dy, None)

        # Place cue objects - these indicate the correct exit
        cue_positions = [(x + 1, y + 1), (x + width - 2, y + 1)]

        for i, pos in enumerate(cue_positions[:len(self.cue_colors)]):
            cue_color = self.cue_colors[i % len(self.cue_colors)]
            cue_obj = Key(cue_color) if i % 2 == 0 else Ball(cue_color)
            self.grid.set(pos[0], pos[1], cue_obj)

    def _create_decision_area(self, x, y, width, height):
        """Create decision area with colored exits"""
        # Clear the area
        for dx in range(width):
            for dy in range(height):
                self.grid.set(x + dx, y + dy, None)

        # Place colored goals as exits
        exit_positions = [
            (x + 1, y + height - 1),
            (x + width - 2, y + height - 1),
            (x + width // 2, y + height - 1)
        ]

        for i, pos in enumerate(exit_positions[:len(self.exit_colors)]):
            exit_color = self.exit_colors[i % len(self.exit_colors)]
            exit_color = self.exit_colors[i]
            self.grid.set(pos[0], pos[1], Goal(color=exit_color))

    def step(self, action):
        self.step_count += 1

        # Track phase transitions based on agent position
        agent_x, agent_y = self.agent_pos

        # Update phase based on position
        if agent_y <= 5 and self.phase == "cue":
            self.phase = "cue"
        elif agent_y > 5 and agent_y < 10 and self.phase in ["cue", "navigate"]:
            self.phase = "navigate"
        elif agent_y >= 10:
            self.phase = "decision"

        # Collect cues in cue phase
        if self.phase == "cue":
            fwd_pos = self.front_pos
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell and hasattr(fwd_cell, 'color') and fwd_cell.color in self.cue_colors:
                if fwd_cell.color not in self.cues_seen:
                    self.cues_seen.append(fwd_cell.color)
                    # Determine correct exit color based on first cue seen
                    if not self.correct_exit_color:
                        self.correct_exit_color = self.cue_mapping.get(fwd_cell.color, "purple")

        obs, reward, terminated, truncated, info = super().step(action)

        # Custom reward logic
        if terminated and hasattr(self, 'goal_reached'):
            goal_obj = self.grid.get(*self.agent_pos)
            if goal_obj and hasattr(goal_obj, 'color'):
                if goal_obj.color == self.correct_exit_color:
                    reward = 1.0  # Correct exit
                else:
                    reward = -0.5  # Wrong exit

        # Add phase and memory info
        info.update({
            'phase': self.phase,
            'cues_seen': self.cues_seen.copy(),
            'correct_exit_color': self.correct_exit_color,
            'steps_in_phase': self._get_steps_in_phase()
        })

        return obs, reward, terminated, truncated, info

    def _get_steps_in_phase(self):
        """Estimate steps spent in current phase"""
        if self.phase == "cue":
            return min(self.step_count, 20)
        elif self.phase == "navigate":
            return max(0, min(self.step_count - 20, 80))
        else:  # decision
            return max(0, self.step_count - 100)

    def set_task_params(self, cue_mapping, maze_layout):
        """Set task parameters for this episode"""
        self.cue_mapping = cue_mapping
        self.maze_layout = int(maze_layout)
        self.correct_exit_color = None
        self.cues_seen = []
        self.phase = "cue"
        self.step_count = 0