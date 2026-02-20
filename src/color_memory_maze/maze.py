import labmaze
import numpy as np
from dm_control.locomotion.walkers import jumping_ball

DEFAULT_CONTROL_TIMESTEP = 0.025
DEFAULT_PHYSICS_TIMESTEP = 0.025

TARGET_COLORS = [
    np.array([170, 38, 30]) / 220,  # red
    np.array([99, 170, 88]) / 220,  # green
    np.array([39, 140, 217]) / 220,  # blue
    np.array([93, 105, 199]) / 220,  # purple
    np.array([220, 193, 59]) / 220,  # yellow
    np.array([220, 128, 107]) / 220,  # salmon
    np.array([255, 165, 0]) / 255,  # orange
    np.array([255, 20, 147]) / 255,  # deep pink
    np.array([0, 255, 255]) / 255,  # cyan
    np.array([128, 0, 128]) / 255,  # purple (darker)
    np.array([255, 255, 255]) / 255,  # white
    np.array([0, 0, 0]) / 255,  # black
    np.array([139, 69, 19]) / 255,  # brown
    np.array([255, 192, 203]) / 255,  # pink
    np.array([128, 128, 128]) / 255,  # gray
    np.array([255, 215, 0]) / 255,  # gold
    np.array([0, 128, 0]) / 255,  # dark green
    np.array([75, 0, 130]) / 255,  # indigo
    np.array([255, 99, 71]) / 255,  # tomato
    np.array([64, 224, 208]) / 255,  # turquoise
]


class RollingBallWithFriction(jumping_ball.RollingBallWithHead):
    def _build(self, roll_damping=5.0, steer_damping=20.0, **kwargs):
        super()._build(**kwargs)
        # Increase friction to the joints, so the movement feels more like traditional
        # first-person navigation control, without much acceleration/deceleration.
        self._mjcf_root.find("joint", "roll").damping = roll_damping
        self._mjcf_root.find("joint", "steer").damping = steer_damping


class TextMazeVaryingWallsFixedLayout(labmaze.FixedMazeWithRandomGoals):
    """Augments standard generated labmaze with some walls marked with different chars."""

    def regenerate(self):
        super().regenerate()
        self._block_variations()

    def _block_variations(self):
        nblocks = 3
        wall_chars = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        n, m = self.entity_layer.shape[:2]
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
                self._change_block_char(i_from, i_to, j_from, j_to, wall_chars[ivar])
                ivar = (ivar + 1) % 10

    def _change_block_char(self, i1, i2, j1, j2, char):
        grid = self.entity_layer
        i, j = np.where(grid[i1:i2, j1:j2] == "*")
        grid[i + i1, j + j1] = char
