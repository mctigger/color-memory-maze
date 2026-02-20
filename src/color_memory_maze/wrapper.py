"""
DrStrategy Maze Environment

Single gymnasium wrapper over composer.Environment that combines observation
remapping, agent position extraction, visual goal generation, goal checking,
and the gymnasium API.
"""

import logging
import math
from contextlib import contextmanager
from typing import Any, List, Tuple

import gymnasium as gym
import numpy as np
from dm_env import specs
from gymnasium import spaces
from gymnasium.utils import seeding

logger = logging.getLogger(__name__)


def _convert_to_space(spec: Any) -> gym.Space:
    """Convert a dm_env spec to a gymnasium space."""
    if isinstance(spec, specs.BoundedArray):
        low = (
            spec.minimum.item()
            if len(spec.minimum.shape) == 0
            else spec.minimum.astype(spec.dtype.type)
        )
        high = (
            spec.maximum.item()
            if len(spec.maximum.shape) == 0
            else spec.maximum.astype(spec.dtype.type)
        )
        return spaces.Box(shape=spec.shape, dtype=spec.dtype.type, low=low, high=high)

    if isinstance(spec, specs.Array) and np.issubdtype(spec.dtype, np.floating):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype.type,
            low=np.full(spec.shape, -np.inf, dtype=spec.dtype.type),
            high=np.full(spec.shape, np.inf, dtype=spec.dtype.type),
        )

    raise ValueError(f"Unexpected spec type {type(spec).__name__}: {spec}")


class GoalManager:
    """Manages goal selection, pre-rendering, and achievement checking."""

    def __init__(
        self,
        goal_positions,
        goal_poses_for_render,
        goal_threshold: float,
        goal_direction_threshold: float,
        random_state: np.random.RandomState,
    ) -> None:
        self._goal_positions = goal_positions
        self._goal_poses_for_render = goal_poses_for_render
        self._threshold = goal_threshold
        self._direction_threshold = goal_direction_threshold
        self._rng = random_state

        self._visual_goals: List[List[np.ndarray]] = []
        self._goal_idx: tuple[int, int] = (0, 0)
        self._goal_pose: np.ndarray = np.zeros(5)

    def prerender(self, render_fn) -> None:
        """Pre-render all goal images. render_fn: (pose) -> np.ndarray."""
        for r in self._goal_poses_for_render:
            room_goals = []
            for g in r:
                room_goals.append(render_fn(g))
            self._visual_goals.append(room_goals)

    def update(self) -> None:
        """Randomly pick a new goal (guaranteed different from current)."""
        current_goal = self._goal_idx
        while True:
            room_idx = self._rng.randint(len(self._goal_positions))
            goal_idx = self._rng.randint(len(self._goal_positions[room_idx]))
            new_goal = (room_idx, goal_idx)
            if new_goal != current_goal:
                break
        self._goal_idx = new_goal
        self._goal_pose = self._goal_positions[room_idx][goal_idx]

    def get_image(self) -> np.ndarray:
        """Return the pre-rendered image for the current goal."""
        return self._visual_goals[self._goal_idx[0]][self._goal_idx[1]]

    def is_achieved(self, position, direction) -> tuple[bool, float]:
        """Return (achieved, L1_distance). Uses scalar math to avoid numpy allocations."""
        gp = self._goal_pose
        diff = abs(float(position[0]) - float(gp[0])) + abs(float(gp[1])) + abs(float(position[1]) - float(gp[2]))

        direction_angle = math.atan2(float(direction[1]), float(direction[0]))
        direction_g = math.atan2(float(gp[4]), float(gp[3]))
        diff_direction = abs((direction_angle - direction_g + math.pi) % (2 * math.pi) - math.pi)

        achieved = (
            diff < self._threshold and diff_direction <= self._direction_threshold
        )
        return achieved, diff


class DrStrategyMazeEnv(gym.Env):
    """Gymnasium env wrapping composer.Environment with DrStrategy visual goals.

    Combines observation remapping, agent position extraction, visual goal
    generation, goal checking, and the gymnasium API in a single wrapper.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    # ── Gymnasium API ──────────────────────────────────────────────────

    def __init__(
        self,
        env,
        layout_config: Any,
        maze_xy_scale: float,
        maze_width: int,
        maze_height: int,
        goal_threshold: float = 0.1,
        goal_direction_threshold: float = np.pi / 4,
        random_state: np.random.RandomState | None = None,
    ) -> None:
        self._env = env

        # Position computation parameters (from AgentPositionWrapper)
        self._maze_xy_scale = maze_xy_scale
        self._center_ji = np.array([maze_width - 2.0, maze_height - 2.0]) / 2.0

        # Layout config validation
        self.layout_config = layout_config
        if not hasattr(layout_config, "goal_poses"):
            raise AttributeError("layout_config must have 'goal_poses' attribute")
        if not hasattr(layout_config, "goal_poses_for_render"):
            raise AttributeError(
                "layout_config must have 'goal_poses_for_render' attribute"
            )

        # Build gymnasium spaces
        self.action_space = _convert_to_space(env.action_spec())

        raw_spec = env.observation_spec()
        obs_spaces = {}
        obs_spaces["image"] = _convert_to_space(raw_spec["walker/egocentric_camera"])
        obs_spaces["target_color"] = _convert_to_space(raw_spec["target_color"])
        obs_spaces["position"] = spaces.Box(-np.inf, np.inf, (2,), np.float64)
        obs_spaces["direction"] = spaces.Box(-np.inf, np.inf, (2,), np.float64)
        obs_spaces["goal_image"] = spaces.Box(0, 255, (64, 64, 3), np.uint8)
        self.observation_space = spaces.Dict(obs_spaces)

        self.np_random, _ = seeding.np_random(None)

        # Initialize goal manager, reset composer env, pre-render goals
        self._goal_manager = GoalManager(
            goal_positions=layout_config.goal_poses,
            goal_poses_for_render=layout_config.goal_poses_for_render,
            goal_threshold=goal_threshold,
            goal_direction_threshold=goal_direction_threshold,
            random_state=(
                random_state if random_state is not None else np.random.RandomState(0)
            ),
        )
        self._env.reset()
        self._goal_manager.prerender(self._render_on_pose)

        # Cache immutable references
        self._task = self._env._task
        self._n_sub_steps = self._env._n_sub_steps
        self._time_limit = self._env._time_limit
        cam_spec = self._env.observation_spec()["walker/egocentric_camera"]
        self._cam_height = cam_spec.shape[0]
        self._cam_width = cam_spec.shape[1]

        # Cache physics-dependent references (re-cached on reset)
        self._cache_physics_refs()

    def _cache_physics_refs(self) -> None:
        """Cache physics-dependent references for fast stepping."""
        physics = self._env._physics
        self._physics = physics
        self._walker_body = physics.bind(self._task._walker.root_body)
        cam_element = self._task._walker.observables.egocentric_camera._mjcf_element
        self._camera_id = physics.model.name2id(
            cam_element.full_identifier, "camera"
        )
        self._target_color_fn = self._task.task_observables[
            "target_color"
        ].observation_callable(physics, np.random.RandomState(0))

    def reset(self, *, seed=None, options=None) -> Tuple[Any, dict]:
        if seed is not None:
            self.np_random, _ = seeding.np_random(seed)
        ts = self._env.reset()
        # Re-cache after reset (physics may have been recompiled)
        self._cache_physics_refs()
        self._goal_manager.update()
        obs = self._extract_obs(ts.observation)
        obs["goal_image"] = self._goal_manager.get_image()
        return obs, {}

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        # Bypass composer.Environment.step() — directly step physics and render.
        # This avoids ~5ms of composer overhead per step (observation updater,
        # target_sphere contact checking on every substep, hooks, etc.).
        physics = self._physics
        physics.set_control(action)
        for _ in range(self._n_sub_steps):
            physics.step()

        # Render camera directly
        image = physics.render(
            height=self._cam_height,
            width=self._cam_width,
            camera_id=self._camera_id,
        )

        # Extract position and orientation directly from physics bindings
        walker_xy = self._walker_body.xpos[:2]
        walker_ji = walker_xy / self._maze_xy_scale + self._center_ji
        orientation = self._walker_body.xmat.reshape(3, 3)[:2, 1]

        obs = {
            "image": image,
            "target_color": self._target_color_fn(),
            "position": walker_ji,
            "direction": orientation,
            "goal_image": self._goal_manager.get_image(),
        }

        is_goal_achieved, distance = self._goal_manager.is_achieved(
            walker_ji, orientation
        )
        reward = 1.0 if is_goal_achieved else 0.0

        if is_goal_achieved:
            self._goal_manager.update()

        # Check time limit for truncation
        truncated = physics.time() >= self._time_limit
        terminated = False
        info = {"success": int(is_goal_achieved), "distance": distance}

        return obs, reward, terminated, truncated, info

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    # ── Observation processing ─────────────────────────────────────────

    def _extract_obs(self, raw_obs: dict) -> dict:
        """Extract and remap observations from raw composer observations."""
        walker_xy = raw_obs["absolute_position"][:2]
        walker_ji = walker_xy / self._maze_xy_scale + self._center_ji
        return {
            "image": raw_obs["walker/egocentric_camera"],
            "target_color": raw_obs["target_color"],
            "position": walker_ji,
            "direction": raw_obs["absolute_orientation"][:2, 1],
        }

    # ── Physics rendering ──────────────────────────────────────────────

    def render_top_down_view(self, resolution=256):
        """Render and return the top-down view of the maze."""
        task = self._env._task

        if not hasattr(task, "observables") or "top_camera" not in task.observables:
            raise AttributeError("Task does not have top_camera observable")

        was_enabled = task.observables["top_camera"].enabled
        original_height = task._maze_arena.observables.top_camera.height
        original_width = task._maze_arena.observables.top_camera.width

        try:
            task.observables["top_camera"].enabled = True
            task._maze_arena.observables.top_camera.height = resolution
            task._maze_arena.observables.top_camera.width = resolution

            random_state = np.random.RandomState()
            top_view = task.observables["top_camera"].observation_callable(
                self._env._physics, random_state
            )()

            if top_view is None:
                raise RuntimeError("Could not retrieve top_camera observation")

            return top_view

        finally:
            task.observables["top_camera"].enabled = was_enabled
            task._maze_arena.observables.top_camera.height = original_height
            task._maze_arena.observables.top_camera.width = original_width

    def _render_on_pose(self, pose: List[float]) -> np.ndarray:
        """Render observation from specified pose."""
        physics = self._env.physics
        with self._state_context_manager(physics):
            physics_state = np.zeros(10)
            physics_state[0] = pose[0]
            physics_state[1] = pose[1]
            physics_state[2] = 0.0
            physics_state[3] = pose[2]

            physics.set_state(physics_state)
            timestep = self._env.step(0)
            image = timestep.observation["walker/egocentric_camera"]

        return image

    @contextmanager
    def _state_context_manager(self, physics):
        saved_state = physics.get_state()
        try:
            yield
        finally:
            physics.set_state(saved_state)
