#!/usr/bin/env python3
from pathlib import Path

import gymnasium as gym
import color_memory_maze  # registers environments
import numpy as np
from PIL import Image


def save_png(image_array, filepath):
    Image.fromarray(
        (image_array * 255).astype(np.uint8)
        if image_array.dtype != np.uint8
        else image_array
    ).save(filepath)


def save_goals(env, goals_dir):
    current_env = env
    while hasattr(current_env, "env"):
        current_env = current_env.env
        if hasattr(current_env, "visual_goals"):
            break
    for room_idx, room_goals in enumerate(current_env.visual_goals):
        for goal_idx, goal_image in enumerate(room_goals):
            save_png(
                goal_image,
                goals_dir
                / f"prerendered_goal_room{room_idx:02d}_goal{goal_idx:02d}.png",
            )


def save_topdown(env, topdown_dir):
    topdown = env.unwrapped.render_top_down_view(resolution=480)
    topdown = np.array(Image.fromarray(topdown).resize((512, 512), Image.LANCZOS))
    save_png(topdown, topdown_dir / "topdown_view.png")


def run_env(env_id):
    env_dir = Path("saved_observations_with_goals") / env_id.replace("-", "_")
    (env_dir / "observations").mkdir(parents=True, exist_ok=True)
    (env_dir / "goals").mkdir(parents=True, exist_ok=True)
    (env_dir / "topdown").mkdir(parents=True, exist_ok=True)

    env = gym.make(env_id, camera_resolution=64)
    obs, info = env.reset()

    save_goals(env, env_dir / "goals")
    save_topdown(env, env_dir / "topdown")
    save_png(obs["image"], env_dir / "observations" / "observation_step_0.png")

    for step in range(1, 51):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if step % 5 == 0:
            save_png(
                obs["image"],
                env_dir / "observations" / f"observation_step_{step:02d}.png",
            )
        if terminated or truncated:
            break

    env.close()


def main():
    for env_id in [
        "MemoryMaze-cmaze-7x7-drstrategy-v0",
        "MemoryMaze-cmaze-15x15-drstrategy-v0",
    ]:
        run_env(env_id)


if __name__ == "__main__":
    main()
