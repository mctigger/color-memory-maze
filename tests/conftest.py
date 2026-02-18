import gymnasium as gym
import pytest

import color_memory_maze  # noqa: F401


@pytest.fixture(scope="module")
def env_7x7():
    """Create a 7x7 DrStrategy environment shared across the module."""
    env = gym.make("MemoryMaze-cmaze-7x7-drstrategy-v0")
    yield env
    env.close()
