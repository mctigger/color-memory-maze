import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import color_memory_maze  # noqa: F401

EXPECTED_OBS_KEYS = {"image", "goal_image", "target_color", "position", "direction"}
REF_DIR = Path(__file__).parent / "reference_data"


# ── Environment creation & observation space ─────────────────────────────


class TestEnvCreation:
    def test_env_creation(self, env_7x7):
        assert env_7x7 is not None

    def test_observation_keys(self, env_7x7):
        obs, _ = env_7x7.reset()
        assert set(obs.keys()) == EXPECTED_OBS_KEYS

    def test_observation_shapes(self, env_7x7):
        obs, _ = env_7x7.reset()
        assert obs["image"].shape == (64, 64, 3)
        assert obs["image"].dtype == np.uint8
        assert obs["goal_image"].shape == (64, 64, 3)
        assert obs["goal_image"].dtype == np.uint8
        assert obs["position"].shape == (2,)
        assert obs["direction"].shape == (2,)
        assert obs["target_color"].shape == (3,)


# ── Environment step ─────────────────────────────────────────────────────


class TestEnvStep:
    def test_step_returns(self, env_7x7):
        env_7x7.reset()
        result = env_7x7.step(np.array([0.0, 0.0]))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_observation_consistency(self, env_7x7):
        obs_reset, _ = env_7x7.reset()
        obs_step, *_ = env_7x7.step(np.array([0.0, 0.0]))
        assert set(obs_step.keys()) == set(obs_reset.keys())
        for key in obs_reset:
            assert obs_step[key].shape == obs_reset[key].shape, (
                f"Shape mismatch for '{key}'"
            )

    def test_action_space(self, env_7x7):
        assert isinstance(env_7x7.action_space, gym.spaces.Box)
        assert env_7x7.action_space.shape == (2,)


# ── Top-down rendering ───────────────────────────────────────────────────


class TestTopDown:
    def test_render_top_down_view(self, env_7x7):
        env_7x7.reset()
        img = env_7x7.unwrapped.render_top_down_view(resolution=64)
        assert isinstance(img, np.ndarray)
        assert img.shape == (64, 64, 3)


# ── Environment lifecycle ────────────────────────────────────────────────


class TestLifecycle:
    def test_multiple_resets(self, env_7x7):
        for _ in range(3):
            obs, info = env_7x7.reset()
            assert "image" in obs

    def test_episode_runs(self, env_7x7):
        env_7x7.reset()
        for _ in range(50):
            obs, reward, terminated, truncated, info = env_7x7.step(
                np.array([-1.0, 0.0])
            )
            if terminated or truncated:
                env_7x7.reset()


# ── 15x15 environment ───────────────────────────────────────────────────


class Test15x15:
    @pytest.fixture(scope="class")
    def env_15x15(self):
        env = gym.make("MemoryMaze-cmaze-15x15-drstrategy-v0")
        yield env
        env.close()

    def test_15x15_env_creation(self, env_15x15):
        obs, _ = env_15x15.reset()
        assert set(obs.keys()) == EXPECTED_OBS_KEYS


# ── FPS benchmark ────────────────────────────────────────────────────────


class TestBenchmark:
    """Verify that the environment can be stepped faster than 200 FPS."""

    NUM_STEPS = 500

    @pytest.fixture(scope="class")
    def bench_env(self):
        env = gym.make("MemoryMaze-cmaze-7x7-drstrategy-v0")
        env.reset()
        yield env
        env.close()

    def test_fps(self, bench_env):
        action = np.array([0.0, 0.0])
        start = time.perf_counter()
        for _ in range(self.NUM_STEPS):
            obs, reward, terminated, truncated, info = bench_env.step(action)
            if terminated or truncated:
                bench_env.reset()
        elapsed = time.perf_counter() - start
        fps = self.NUM_STEPS / elapsed
        assert fps > 200, f"FPS {fps:.1f} is below the 200 FPS target"


# ── Rendering snapshot regression ────────────────────────────────────────
#
# Each class creates a fresh environment and resets it exactly once.
# All observation snapshot tests use that single reset result so the
# seeded random state is in a known position.


class TestSnapshotRegression7x7:
    """Verify rendered observations match saved reference data."""

    @pytest.fixture(scope="class")
    def fresh_7x7(self):
        env = gym.make("MemoryMaze-cmaze-7x7-drstrategy-v0")
        obs, info = env.reset()
        top_down = env.unwrapped.render_top_down_view(resolution=64)
        yield env, obs, top_down
        env.close()

    def test_reset_image(self, fresh_7x7):
        _, obs, _ = fresh_7x7
        ref = np.load(REF_DIR / "7x7_reset_image.npy")
        np.testing.assert_allclose(obs["image"], ref, atol=2)

    def test_reset_goal_image(self, fresh_7x7):
        _, obs, _ = fresh_7x7
        ref = np.load(REF_DIR / "7x7_reset_goal_image.npy")
        np.testing.assert_allclose(obs["goal_image"], ref, atol=2)

    def test_reset_target_color(self, fresh_7x7):
        _, obs, _ = fresh_7x7
        ref = np.load(REF_DIR / "7x7_reset_target_color.npy")
        np.testing.assert_allclose(obs["target_color"], ref, atol=1e-5)

    def test_reset_top_down(self, fresh_7x7):
        _, _, top_down = fresh_7x7
        ref = np.load(REF_DIR / "7x7_reset_top_down.npy")
        np.testing.assert_allclose(top_down, ref, atol=2)

    def test_reset_position(self, fresh_7x7):
        _, obs, _ = fresh_7x7
        ref = np.load(REF_DIR / "7x7_reset_position.npy")
        np.testing.assert_allclose(obs["position"], ref, atol=1e-5)

    def test_reset_direction(self, fresh_7x7):
        _, obs, _ = fresh_7x7
        ref = np.load(REF_DIR / "7x7_reset_direction.npy")
        np.testing.assert_allclose(obs["direction"], ref, atol=1e-5)


class TestTimeLimitCustomization:
    """Tests for the time_limit parameter in gym.make."""

    def test_custom_time_limit_truncates_episode(self):
        # time_limit=2s, control_timestep=0.25s → truncation at step 8
        env = gym.make("MemoryMaze-cmaze-7x7-drstrategy-v0", time_limit=2)
        env.reset()
        seen_truncated = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
            if truncated:
                seen_truncated = True
                assert not terminated
                break
        env.close()
        assert seen_truncated, "Episode should have truncated within 10 steps"

    def test_non_episodic_does_not_truncate(self):
        # time_limit=-1 → np.inf; should not truncate where time_limit=2 would
        env = gym.make("MemoryMaze-cmaze-7x7-drstrategy-v0", time_limit=-1)
        env.reset()
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
            assert not truncated, "Non-episodic env must not truncate"
            assert not terminated, "Non-episodic env must not terminate"
        env.close()

    def test_custom_time_limit_15x15(self):
        env = gym.make("MemoryMaze-cmaze-15x15-drstrategy-v0", time_limit=2)
        env.reset()
        seen_truncated = False
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
            if truncated:
                seen_truncated = True
                break
        env.close()
        assert seen_truncated, "15x15 episode should have truncated within 10 steps"


class TestSnapshotRegression15x15:
    """Verify 15x15 rendered observations match saved reference data."""

    @pytest.fixture(scope="class")
    def fresh_15x15(self):
        env = gym.make("MemoryMaze-cmaze-15x15-drstrategy-v0")
        obs, info = env.reset()
        yield env, obs
        env.close()

    def test_reset_image(self, fresh_15x15):
        _, obs = fresh_15x15
        ref = np.load(REF_DIR / "15x15_reset_image.npy")
        np.testing.assert_allclose(obs["image"], ref, atol=2)

    def test_reset_goal_image(self, fresh_15x15):
        _, obs = fresh_15x15
        ref = np.load(REF_DIR / "15x15_reset_goal_image.npy")
        np.testing.assert_allclose(obs["goal_image"], ref, atol=2)

