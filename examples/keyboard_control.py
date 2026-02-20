#!/usr/bin/env python3
"""
Interactive navigation script for a specified MemoryMaze environment.

Displays current observation, goal image, and top-down view side by side.
Use WASD keys to navigate the agent through the maze.

Usage: python navigate_consistent_target.py --env_id <env_id>

Controls:
- W/w: Move forward
- A/a: Turn left
- D/d: Turn right
- S/s: No-op (no backward available)
- Q/q: Quit
- R/r: Reset environment
"""

import argparse
import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

import color_memory_maze  # noqa: F401


class ConsistentTargetNavigator:
    def __init__(self, env_id, camera_resolution=64, different_floor_textures=True):
        self.env_id = env_id
        self.env = gym.make(
            env_id,
            camera_resolution=camera_resolution,
            different_floor_textures=different_floor_textures,
        )
        self.obs, self.info = self.env.reset()

        # Create a separate environment for top-down view using the approach from generate_maze_png.py
        # Set up the figure with subplots
        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 3, figsize=(15, 5))
        self.fig.suptitle(f"Interactive Navigation: {env_id}")

        # Configure axes
        self.ax1.set_title("Current Observation")
        self.ax1.axis("off")

        self.ax2.set_title("Goal Image")
        self.ax2.axis("off")

        self.ax3.set_title("Top-down View")
        self.ax3.axis("off")

        # Initialize image displays
        self.img1 = self.ax1.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
        self.img2 = self.ax2.imshow(np.zeros((64, 64, 3), dtype=np.uint8))
        self.img3 = self.ax3.imshow(np.zeros((64, 64, 3), dtype=np.uint8))

        # Add control instructions
        self.fig.text(
            0.02,
            0.02,
            "Controls: W=Forward, A=Turn Left, D=Turn Right, S=No-op, R=Reset, Q=Quit",
            fontsize=10,
            ha="left",
        )

        # Connect keyboard events
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Initial display
        self.update_display()

    def on_key_press(self, event):
        """Handle keyboard input for navigation"""
        if event.key is None:
            return

        key = event.key.lower()

        if key == "q":
            plt.close("all")
            self.close()
            sys.exit(0)
        elif key == "r":
            self.reset_environment()
        else:
            action = self.get_action_from_key(key)
            if action is not None:
                self.step_environment(action)

    def get_action_from_key(self, key):
        """Convert key press to environment action"""
        action_map = {
            "w": np.array([-0.5, 0.0]),  # Move forward
            "s": np.array([0.0, 0.0]),  # No-op
            "a": np.array([0.0, -0.5]),  # Turn left
            "d": np.array([0.0, +0.5]),  # Turn right
        }
        return action_map.get(key)

    def step_environment(self, action):
        """Take a step in the environment and update display"""
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)

        distance = self.info.get("distance", float("nan"))
        print(f"Step | Dist to goal: {distance:.3f} | Reward: {reward}")

        self.update_display()

        if terminated or truncated:
            print(f"Episode ended! Reward: {reward}")
            if terminated:
                print("Goal reached!" if reward > 0 else "Episode terminated")

    def reset_environment(self):
        """Reset the environment"""
        try:
            self.obs, self.info = self.env.reset()

            # Also reset the top-down environment
            self.update_display()
            print("Environment reset")
        except Exception as e:
            print(f"Error resetting environment: {e}")

    def update_display(self):
        """Update all three image displays"""
        # Current observation (first-person view)
        current_obs = self.obs["image"]

        self.img1.set_array(current_obs)

        # Goal image - try to get from info or environment
        goal_image = self.obs["goal_image"]
        self.img2.set_array(goal_image)

        # Top-down view - try to get from environment
        topdown_view = self.get_topdown_view()
        self.img3.set_array(topdown_view)

        # Update step counter and distance if available
        step_count = getattr(self.env, "_step_count", 0)
        distance = self.info.get("distance", float("nan"))
        self.fig.suptitle(f"{self.env_id} (Step: {step_count}, Dist to goal: {distance:.2f})")

        plt.draw()

    def get_topdown_view(self):
        """Get top-down view using built-in render_top_down_view method"""
        return self.env.unwrapped.render_top_down_view(resolution=64)

    def render_maze_layout(self, obs):
        """Render top-down view from maze layout similar to DrawMinimapWrapper"""
        try:
            from PIL import Image

            maze = obs["maze_layout"]
            N = maze.shape[0]
            SIZE = 64  # Fixed size for display

            # Draw map - similar to DrawMinimapWrapper logic
            map_img = np.zeros((N, N, 3), np.uint8)  # walls in black
            map_img[:, :] += (maze == 1)[..., None] * np.array(
                [[[255, 255, 255]]], np.uint8
            )  # corridors in white
            map_img[:, :] += (maze == 2)[..., None] * np.array(
                [[[0, 255, 0]]], np.uint8
            )  # path in green

            # Add agent position if available
            if "agent_pos" in obs:
                x, y = obs["agent_pos"]
                if 0 <= int(y) < N and 0 <= int(x) < N:
                    map_img[int(y), int(x)] = np.array(
                        [255, 0, 0], np.uint8
                    )  # agent in red

            # Add target positions if available (from info or obs)
            if hasattr(self, "info") and "target_pos" in self.info:
                tx, ty = self.info["target_pos"]
                if 0 <= int(ty) < N and 0 <= int(tx) < N:
                    map_img[int(ty), int(tx)] = np.array(
                        [0, 0, 255], np.uint8
                    )  # target in blue

            # Flip vertically for proper orientation
            map_img = np.flip(map_img, 0)

            # Scale to desired size
            if N != SIZE:
                mapimg = Image.fromarray(map_img)
                mapimg = mapimg.resize((SIZE, SIZE), Image.Resampling.NEAREST)
                map_img = np.array(mapimg)

            return map_img

        except Exception as e:
            print(f"Error rendering maze layout: {e}")
            return self.create_simple_topdown()

    def create_simple_topdown(self):
        """Create a simple top-down maze representation"""
        # Create a 7x7 grid representation
        grid_size = 64
        cell_size = grid_size // 7
        topdown = (
            np.ones((grid_size, grid_size, 3), dtype=np.uint8) * 128
        )  # Gray background

        # Add grid lines
        for i in range(8):
            pos = i * cell_size
            topdown[pos : pos + 2, :] = [0, 0, 0]  # Horizontal lines
            topdown[:, pos : pos + 2] = [0, 0, 0]  # Vertical lines

        # Try to mark agent position if available
        try:
            if hasattr(self.env.unwrapped, "_agent_pos"):
                agent_pos = self.env.unwrapped._agent_pos
                y, x = int(agent_pos[0] * cell_size), int(agent_pos[1] * cell_size)
                topdown[y : y + cell_size // 2, x : x + cell_size // 2] = [
                    0,
                    255,
                    0,
                ]  # Green for agent
        except:
            pass

        return topdown

    def run(self):
        """Start the interactive navigation"""
        print("Interactive Navigation Started!")
        print(
            "Controls: W=Forward, A=Turn Left, D=Turn Right, S=No-op, R=Reset, Q=Quit"
        )
        print("Click on the window and use keyboard to control the agent.")

        plt.tight_layout()
        plt.show()

    def close(self):
        """Clean up environments"""
        if hasattr(self, "env"):
            self.env.close()
        if hasattr(self, "topdown_env") and self.topdown_env is not None:
            self.topdown_env.close()


def main():
    """Main function to run the navigation example"""
    parser = argparse.ArgumentParser(
        description="Interactive navigation for MemoryMaze environments"
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="MemoryMaze-cmaze-7x7-drstrategy-v0",
        help="Environment ID to use (default: MemoryMaze-cmaze-7x7-drstrategy-v0)",
    )
    parser.add_argument(
        "--camera_resolution",
        type=int,
        default=64,
        help="Camera resolution for environment rendering (default: 64)",
    )
    parser.add_argument(
        "--no_different_floor_textures",
        action="store_true",
        help="Disable different floor textures (use single blue floor)",
    )
    args = parser.parse_args()

    navigator = None
    try:
        navigator = ConsistentTargetNavigator(
            args.env_id, args.camera_resolution, not args.no_different_floor_textures
        )
        navigator.run()
    except KeyboardInterrupt:
        print("\nNavigation interrupted by user")
        raise
    except Exception as e:
        print(f"Error running navigation: {e}")
        raise
    finally:
        if navigator is not None:
            navigator.close()


if __name__ == "__main__":
    main()
