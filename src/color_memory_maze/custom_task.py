from typing import Any, Type

import numpy as np
from dm_control import composer

from color_memory_maze.maze import (
    InertRollingBallWithFriction,
    InertTargetSphere,
)
from color_memory_maze.maze_layouts import Maze7x7, Maze15x15
from color_memory_maze.skybox import FixedSkyBox
from color_memory_maze.unified_maze_arena import MazeConfig, UnifiedMazeWithTargetsArena
from color_memory_maze.unified_task import TaskConfig, UnifiedMemoryMazeTask
from color_memory_maze.wrapper import DrStrategyMazeEnv

DEFAULT_CONTROL_FREQ = 4.0


def _create_drstrategy_env(
    n_targets: int,
    time_limit: float,
    layout_config_class: Type[Any],
    **kwargs: Any,
) -> Any:
    """Create a DrStrategy environment with visual target generation."""
    goal_threshold = kwargs.pop("goal_threshold", 0.1)
    goal_direction_threshold = kwargs.pop("goal_direction_threshold", np.pi / 4)
    control_freq = kwargs.pop("control_freq", DEFAULT_CONTROL_FREQ)
    camera_resolution = kwargs.pop("camera_resolution", 64)
    seed = kwargs.pop("seed", 42)

    layout_config = layout_config_class()
    random_state = np.random.RandomState(seed)
    agent = InertRollingBallWithFriction(camera_height=0.3)

    maze_config = MazeConfig(
        xy_scale=2.0,
        z_height=1.5,
        target_radius=0.6,
        target_height_above_ground=-0.4,
        control_timestep=1.0 / control_freq,
    )

    arena = UnifiedMazeWithTargetsArena(
        entity_layer=layout_config.layout,
        num_objects=n_targets,
        random_state=random_state,
        config=maze_config,
        skybox_texture=FixedSkyBox("sky_03"),
        aesthetic="default",
        name="custom_maze",
    )

    task_config = TaskConfig(
        randomize_spawn_position=False,
        randomize_spawn_rotation=False,
        shuffle_target_positions=False,
        allow_same_color_targets=False,
        n_targets=n_targets,
        target_reward_scale=1.0,
        enable_global_task_observables=True,
        camera_resolution=camera_resolution,
        target_class=InertTargetSphere,
        disable_unused_observables=True,
    )

    task = UnifiedMemoryMazeTask(
        walker=agent,
        maze_arena=arena,
        task_config=task_config,
        maze_config=maze_config,
    )

    effective_time_limit = np.inf if time_limit == -1 else time_limit - 1e-3
    composer_env = composer.Environment(
        time_limit=effective_time_limit,
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )

    env = DrStrategyMazeEnv(
        env=composer_env,
        layout_config=layout_config,
        maze_xy_scale=task._maze_arena.xy_scale,
        maze_width=task._maze_arena.maze.width,
        maze_height=task._maze_arena.maze.height,
        goal_threshold=goal_threshold,
        goal_direction_threshold=goal_direction_threshold,
        random_state=np.random.RandomState(seed),
    )

    return env


def memory_maze_cmaze_7x7_drstrategy(**kwargs: Any) -> Any:
    """DrStrategy 7x7 complex maze with visual target generation."""
    time_limit = kwargs.pop("time_limit", 500)
    return _create_drstrategy_env(5, time_limit, Maze7x7, **kwargs)


def memory_maze_cmaze_15x15_drstrategy(**kwargs: Any) -> Any:
    """DrStrategy 15x15 complex maze with visual target generation."""
    time_limit = kwargs.pop("time_limit", 1000)
    return _create_drstrategy_env(8, time_limit, Maze15x15, **kwargs)
