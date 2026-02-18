"""Unified maze arena implementation.

This module provides the maze arena class for the DrStrategy memory maze environments.
"""

import string
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from dm_control.locomotion.arenas import covering, labmaze_textures, mazes
from numpy.random import RandomState

from color_memory_maze.maze import (
    DEFAULT_CONTROL_TIMESTEP,
    DEFAULT_PHYSICS_TIMESTEP,
    TextMazeVaryingWallsFixedLayout,
)


@dataclass
class MazeConfig:
    """Configuration parameters for the maze environment."""

    xy_scale: float = 2.0
    z_height: float = 2.0
    target_radius: float = 0.3
    target_height_above_ground: float = 0.0
    physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP
    control_timestep: float = DEFAULT_CONTROL_TIMESTEP


def _create_colorful_wall_textures():
    """Create colorful wall textures using the tab20 colormap."""
    cmap = plt.get_cmap("tab20")

    class ColorfulWallTextures(labmaze_textures.WallTextures):
        def _build(self, style="style_01"):
            super()._build(style)
            if hasattr(self, "_color"):
                self._textures = [
                    self._mjcf_root.asset.add(
                        "texture",
                        type="2d",
                        name="wall",
                        builtin="flat",
                        rgb1=self._color,
                        width=100,
                        height=100,
                    )
                ]

        def __init__(self, color=[0.8, 0.8, 0.8], model="style_01"):
            self._color = color
            super().__init__(model)

    wall_textures = {"*": ColorfulWallTextures([0.8, 0.8, 0.8])}
    for index in range(10):
        wall_textures[str(index)] = ColorfulWallTextures(cmap(index * 2)[:3])
    for index, idx1 in enumerate([i for i in range(1, 20, 2)]):
        wall_textures[str(index + 10)] = ColorfulWallTextures(cmap(idx1)[:3])

    return wall_textures


def _create_colorful_floor_textures():
    """Create colorful floor textures using the tab20 colormap."""
    floor_colors = []
    cmap_floor = plt.get_cmap("tab20")
    for index in range(10):
        floor_colors.append([i for i in cmap_floor(index * 2)[:3]])
    for index, idx1 in enumerate([i for i in range(1, 20, 2)]):
        floor_colors.append([i for i in cmap_floor(idx1)[:3]])

    class ColorfulFloorTextures(labmaze_textures.FloorTextures):
        def _build(self, style):
            super()._build(style)
            if hasattr(self, "_colors"):
                self._textures = []
                for i, color in enumerate(self._colors):
                    self._textures.append(
                        self._mjcf_root.asset.add(
                            "texture",
                            type="2d",
                            name="floor" + str(i),
                            builtin="flat",
                            rgb1=color,
                            width=100,
                            height=100,
                        )
                    )

        def __init__(self, style, colors):
            self._colors = colors
            super().__init__(style)

    return ColorfulFloorTextures("style_01", floor_colors)


class UnifiedMazeWithTargetsArena(mazes.MazeWithTargets):
    """Maze arena with colorful wall and floor textures for DrStrategy environments."""

    def __init__(
        self,
        entity_layer: str,
        num_objects: int,
        random_state: RandomState,
        config: Optional[MazeConfig] = None,
        skybox_texture=None,
        aesthetic="default",
        name="unified_maze",
    ):
        self._config = config or MazeConfig()

        maze = TextMazeVaryingWallsFixedLayout(
            entity_layer=entity_layer,
            num_spawns=1,
            num_objects=num_objects,
            random_state=random_state,
        )

        wall_textures = _create_colorful_wall_textures()
        floor_textures = _create_colorful_floor_textures()

        super().__init__(
            maze=maze,
            xy_scale=self._config.xy_scale,
            z_height=self._config.z_height,
            skybox_texture=skybox_texture,
            wall_textures=wall_textures,
            floor_textures=floor_textures,
            aesthetic=aesthetic,
            name=name,
        )

        self._setup_floor_block_variations()

    def regenerate(self, random_state: Optional[RandomState] = None):
        """Regenerate the maze layout."""
        self._maze.regenerate()
        self._find_spawn_and_target_positions()

        if self._text_maze_regenerated_hook:
            self._text_maze_regenerated_hook()

        # Clean up old geometries
        for geom_name in self._texturing_geom_names:
            del self._mjcf_root.worldbody.geom[geom_name]
        self._texturing_geom_names = []

        for material_name in self._texturing_material_names:
            del self._mjcf_root.asset.material[material_name]
        self._texturing_material_names = []

        self._maze_body.geom.clear()

        # Apply wall textures (deterministic: always use first texture)
        if random_state is None:
            random_state = np.random.RandomState()
        self._current_wall_texture = {
            wall_char: wall_texture_list[0]
            for wall_char, wall_texture_list in self._wall_textures.items()
        }
        for wall_char in self._wall_textures:
            self._make_wall_geoms(wall_char)

        # Setup floor variations and apply them
        self._setup_floor_block_variations()
        self._make_floor_variations()

    def _setup_floor_block_variations(self):
        """Create block-based floor variations using ASCII chars."""
        nblocks = 3
        _DEFAULT_FLOOR_CHAR = "."
        floor_chars = _DEFAULT_FLOOR_CHAR + string.ascii_uppercase

        n, m = self._maze.variations_layer.shape[:2]
        mblocks = m * nblocks // n
        if mblocks <= 1:
            mblocks = 1
        elif 10 % mblocks == 0:
            mblocks -= 1

        ivar = 0
        for i in range(nblocks):
            for j in range(mblocks):
                i_from = i * n // nblocks
                i_to = (i + 1) * n // nblocks
                j_from = j * m // mblocks
                j_to = (j + 1) * m // mblocks
                grid = self._maze.variations_layer
                ii, jj = np.where(grid[i_from:i_to, j_from:j_to] == ".")
                grid[ii + i_from, jj + j_from] = floor_chars[ivar]
                ivar = (ivar + 1) % 10

    def _make_floor_variations(self, build_tile_geoms_fn=None):
        """Generate floor tile geometries with colorful textures."""
        _DEFAULT_FLOOR_CHAR = "."

        main_floor_texture = self._floor_textures[0]
        if len(self._floor_textures) > 1:
            room_floor_textures = self._floor_textures[1:]
        else:
            room_floor_textures = [main_floor_texture]

        possible_variations = _DEFAULT_FLOOR_CHAR + string.ascii_uppercase
        for i_var, variation in enumerate(possible_variations):
            if variation not in self._maze.variations_layer:
                break

            if build_tile_geoms_fn is None:
                tiles = covering.make_walls(
                    self._maze.variations_layer,
                    wall_char=variation,
                    make_odd_sized_walls=True,
                )
            else:
                tiles = build_tile_geoms_fn(wall_char=variation)

            if variation == _DEFAULT_FLOOR_CHAR:
                variation_texture = main_floor_texture
            else:
                texture_idx = i_var % len(room_floor_textures)
                variation_texture = room_floor_textures[texture_idx]

            for i, tile in enumerate(tiles):
                tile_mid = covering.GridCoordinates(
                    (tile.start.y + tile.end.y - 1) / 2,
                    (tile.start.x + tile.end.x - 1) / 2,
                )
                tile_pos = np.array(
                    [
                        (tile_mid.x - self._x_offset) * self._xy_scale,
                        -(tile_mid.y - self._y_offset) * self._xy_scale,
                        0.0,
                    ]
                )
                tile_size = np.array(
                    [
                        (tile.end.x - tile_mid.x - 0.5) * self._xy_scale,
                        (tile.end.y - tile_mid.y - 0.5) * self._xy_scale,
                        self._xy_scale,
                    ]
                )

                if variation == _DEFAULT_FLOOR_CHAR:
                    tile_name = f"floor_{i}"
                else:
                    tile_name = f"floor_{variation}_{i}"

                self._tile_geom_names[tile.start] = tile_name
                self._texturing_material_names.append(tile_name)
                self._texturing_geom_names.append(tile_name)

                material = self._mjcf_root.asset.add(
                    "material",
                    name=tile_name,
                    texture=variation_texture,
                    texrepeat=(2 * tile_size[[0, 1]] / self._xy_scale),
                )
                self._mjcf_root.worldbody.add(
                    "geom",
                    name=tile_name,
                    type="plane",
                    material=material,
                    pos=tile_pos,
                    size=tile_size,
                    contype=0,
                    conaffinity=0,
                )
