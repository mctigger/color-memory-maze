"""Unified memory maze task implementation.

This module provides a single, configurable task class that replaces both
MemoryMazeTask and CMemoryMazeTask, eliminating code duplication while
maintaining backward compatibility through factory methods.

Design principles:
- Configuration over inheritance: Use dataclasses for behavior specification
- Single responsibility: Each method has one clear purpose
- Fail fast: Validate configurations early in the constructor
- Composition over inheritance: Delegate arena-specific logic to arenas
"""

import functools
from dataclasses import dataclass
from typing import Optional

import numpy as np
from dm_control import mjcf
from dm_control.composer.observation import observable as observable_lib
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import random_goal_maze
from numpy.random import RandomState

from color_memory_maze.maze import (
    TARGET_COLORS,
)
from color_memory_maze.unified_maze_arena import MazeConfig


@dataclass
class TaskConfig:
    """Configuration for maze task behavior.

    Encapsulates all behavioral differences between task variants,
    making the distinction between original and custom tasks explicit.
    """

    # Spawn behavior: True enables randomization for training variety
    randomize_spawn_position: bool = True
    randomize_spawn_rotation: bool = True

    # Target placement: True shuffles positions for training robustness
    shuffle_target_positions: bool = True

    # Color management: True allows color repetition for many-target scenarios
    allow_same_color_targets: bool = False

    # Task parameters
    n_targets: int = 3
    target_reward_scale: float = 1.0
    enable_global_task_observables: bool = False
    camera_resolution: int = 64

class UnifiedMemoryMazeTask(random_goal_maze.NullGoalMaze):
    """Unified memory maze task supporting both original and custom behaviors.

    Consolidates MemoryMazeTask and CMemoryMazeTask functionality through
    configuration-driven design, eliminating code duplication while
    maintaining full backward compatibility.
    """

    def __init__(
        self,
        walker,
        maze_arena,
        task_config: Optional[TaskConfig] = None,
        maze_config: Optional[MazeConfig] = None,
    ):
        """Initialize unified memory maze task.

        Args:
            task_config: Task behavior configuration. Defaults to original task behavior.
            maze_config: Maze configuration parameters. If None, uses default values.
        """
        self._task_config = task_config or TaskConfig()
        self._maze_config = maze_config or MazeConfig()

        super().__init__(
            walker=walker,
            maze_arena=maze_arena,
            randomize_spawn_position=self._task_config.randomize_spawn_position,
            randomize_spawn_rotation=self._task_config.randomize_spawn_rotation,
            contact_termination=False,
            enable_global_task_observables=self._task_config.enable_global_task_observables,
            physics_timestep=self._maze_config.physics_timestep,
            control_timestep=self._maze_config.control_timestep,
        )

        # Task state management
        self._target_reward_scale = self._task_config.target_reward_scale
        self._current_target_ix = 0
        self._rewarded_this_step = False
        self._targets_obtained = 0

        # Initialize targets with color management
        self._targets = self._create_targets(
            self._task_config.n_targets,
            self._maze_config.target_radius,
            self._maze_config.target_height_above_ground,
        )

        # Setup observables if requested
        if self._task_config.enable_global_task_observables:
            self._setup_target_observables(walker, self._task_config.n_targets)

        # Configure task observables
        self._setup_task_observables()

        # Set camera resolution
        self._configure_cameras(self._task_config.camera_resolution)

    def _create_targets(
        self, n_targets: int, radius: float, height_above_ground: float
    ) -> list:
        """Create target spheres with appropriate color assignment.

        Handles both distinct colors (original) and repeating colors (extended) modes.
        """
        targets = []

        for i in range(n_targets):
            color = self._get_target_color(i)
            target = target_sphere.TargetSphere(
                radius=radius,
                height_above_ground=radius + height_above_ground,
                rgb1=tuple(color * 1.0),
                rgb2=tuple(color * 1.0),
            )
            targets.append(target)
            self._maze_arena.attach(target)

        return targets

    def _get_target_color(self, target_index: int) -> np.ndarray:
        """Get color for target at given index, handling color cycling if enabled."""
        if self._task_config.allow_same_color_targets:
            return TARGET_COLORS[target_index % len(TARGET_COLORS)]
        return TARGET_COLORS[target_index]

    def _setup_target_observables(self, walker, n_targets: int) -> None:
        """Configure egocentric target position observables."""

        def xpos_origin_callable(phys):
            return phys.bind(walker.root_body).xpos

        def _target_pos(physics, target):
            return physics.bind(target.geom).xpos

        for i in range(n_targets):
            # Absolute target position
            walker.observables.add_observable(
                f"target_abs_{i}",
                observable_lib.Generic(
                    functools.partial(_target_pos, target=self._targets[i])
                ),
            )
            # Relative target position (egocentric)
            walker.observables.add_egocentric_vector(
                f"target_rel_{i}",
                observable_lib.Generic(
                    functools.partial(_target_pos, target=self._targets[i])
                ),
                origin_callable=xpos_origin_callable,
            )

    def _setup_task_observables(self) -> None:
        """Configure task-level observables for current target tracking."""
        self._task_observables = super().task_observables

        # Current target index observable
        self._task_observables["target_index"] = observable_lib.Generic(
            lambda _: self._current_target_ix
        )
        self._task_observables["target_index"].enabled = True

        # Current target color observable
        self._task_observables["target_color"] = observable_lib.Generic(
            lambda _: self._get_target_color(self._current_target_ix)
        )
        self._task_observables["target_color"].enabled = True

    def _configure_cameras(self, resolution: int) -> None:
        """Set camera resolution for both egocentric and top-down views."""
        self._walker.observables.egocentric_camera.height = resolution
        self._walker.observables.egocentric_camera.width = resolution
        self._maze_arena.observables.top_camera.height = resolution
        self._maze_arena.observables.top_camera.width = resolution

    @property
    def task_observables(self):
        return self._task_observables

    @property
    def name(self):
        return "goal_maze"

    def initialize_episode_mjcf(self, unused_random_state: RandomState):
        """Initialize episode with maze regeneration and target placement."""
        # Regenerate maze layout
        self._maze_arena.regenerate(unused_random_state)

        # Ensure targets can be placed, regenerating maze if necessary
        while not self._place_targets(unused_random_state):
            self._maze_arena.regenerate(unused_random_state)

        # Select initial target
        self._pick_new_target(unused_random_state)

    def initialize_episode(self, physics, random_state: RandomState):
        """Reset episode state."""
        super().initialize_episode(physics, random_state)
        self._rewarded_this_step = False
        self._targets_obtained = 0

    def after_step(self, physics, random_state: RandomState):
        """Handle target collection and target switching."""
        super().after_step(physics, random_state)
        self._rewarded_this_step = False

        for i, target in enumerate(self._targets):
            if target.activated:
                # Reward only for collecting the current target
                if i == self._current_target_ix:
                    self._rewarded_this_step = True
                    self._targets_obtained += 1
                    self._pick_new_target(random_state)

                # Reset target activation state
                target.reset(physics)

    def should_terminate_episode(self, physics):
        """Delegate termination decision to parent class."""
        return super().should_terminate_episode(physics)

    def get_reward(self, physics):
        """Provide sparse reward signal for target collection."""
        del physics  # Unused parameter
        return self._target_reward_scale if self._rewarded_this_step else 0.0

    def _place_targets(self, rng: RandomState) -> bool:
        """Place targets in available positions with optional shuffling.

        Returns:
            True if all targets were successfully placed, False if regeneration needed.
        """
        possible_positions = list(self._maze_arena.target_positions)

        # Apply position shuffling based on configuration
        if self._task_config.shuffle_target_positions:
            rng.shuffle(possible_positions)

        # Ensure sufficient positions available
        if len(possible_positions) < len(self._targets):
            return False

        # Assign positions to targets
        for target, pos in zip(self._targets, possible_positions):
            mjcf.get_attachment_frame(target.mjcf_model).pos = pos

        return True

    def _pick_new_target(self, rng: RandomState) -> None:
        """Select next target, avoiding currently activated targets."""
        while True:
            ix = rng.randint(len(self._targets))
            if not self._targets[ix].activated:
                self._current_target_ix = ix
                break
