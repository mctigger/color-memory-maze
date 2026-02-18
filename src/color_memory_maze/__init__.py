import os

import color_memory_maze.custom_task

# NOTE: Env MUJOCO_GL=egl is necessary for headless hardware rendering on GPU,
# but breaks when running on a CPU machine. Alternatively set MUJOCO_GL=osmesa.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

# Register gym environments, if gym is available
from functools import partial as f

from gymnasium.envs.registration import register

# Register 7x7 maze with DrStrategy visual targets (returns observation dictionary)
register(
    id="MemoryMaze-cmaze-7x7-drstrategy-v0",
    entry_point=f(
        color_memory_maze.custom_task.memory_maze_cmaze_7x7_drstrategy,
    ),
    kwargs={},
)

# Register 15x15 maze with DrStrategy visual targets
register(
    id="MemoryMaze-cmaze-15x15-drstrategy-v0",
    entry_point=f(
        color_memory_maze.custom_task.memory_maze_cmaze_15x15_drstrategy,
    ),
    kwargs={},
)
