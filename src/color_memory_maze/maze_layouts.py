CMaze_7x7_LAYOUT = """
*********
*P  *G  *
*** *** *
*G*     *
* *** ***
*     *G*
* ***   *
* G*G   *
*********
"""[1:]

CMaze_15x15_LAYOUT = """
*****************
***P      *     *
*** *  G* *   * *
*       * *  G* *
*   * *** ***** *
*         *     *
* ***   * * *** *
*G      *   *   *
*** *** * * *  G*
*** *     * *   *
*** * *   * * * *
*     *G      * *
* *** * * *  G* *
*  G      *   * *
*   *   * *** * *
*   *  G*       *
*****************
"""[1:]


class Layout:
    def __init__(self) -> None:
        self.layout = None
        self.rooms = None
        self.min_x, self.max_x, self.min_y, self.max_y = None, None, None, None
        self.invert_origin = None
        self.max_num_steps = 500
        self.len_x = None
        self.len_y = None

    def get_rooms(self):
        return self.rooms

    def get_min_max_coords(self, shape):
        self.min_x = -0.5 - 1
        self.max_x = -0.5 + shape[1] + 2
        self.min_y = -0.5 - 1
        self.max_y = -0.5 + shape[0] + 2
        return self.min_x, self.max_x, self.min_y, self.max_y

    def cut_topdown_view(self, topdown_view):
        return topdown_view


class Maze7x7(Layout):
    def __init__(self) -> None:
        super().__init__()
        self.layout = CMaze_7x7_LAYOUT
        self.invert_origin = lambda p: [
            p[0],
            -p[1] + 7,
        ]
        self.goal_poses_for_render = [
            [[-6, 0.2, -1.57079633]],  # green
            [[3.8, 6, 3.14159265]],  # red
            [[-5.8, -6, 0], [1.8, -6, 3.14159265]],  # pruple, yellow
            [[6, -3.8, -1.57079633]],  # blue
        ]
        self.goal_poses = [
            [[0.5, 0.0, 3.6, 0.0, 1.0]],
            [[5.4, 0.0, 6.5, -1.0, -3.5897931e-09]],
            [[0.6, 0.0, 0.5, 1.0, 0.0], [4.4, 0.0, 0.5, -1.0, -3.5897931e-09]],
            [[6.5, 0.0, 1.6, 0.0, 1.0]],
        ]


class Maze15x15(Layout):
    def __init__(self) -> None:
        super().__init__()
        self.layout = CMaze_15x15_LAYOUT
        self.rooms = [
            [-1.5, 8.5, -1.5, 8.5],
            [8.5, 16.5, -1.5, 8.5],
            [-1.5, 8.5, 8.5, 16.5],
            [8.5, 16.5, 8.5, 16.5],
        ]
        self.invert_origin = lambda p: [p[0], -p[1] + 15]
        self.max_num_steps = 1000
        self.len_x = 15
        self.len_y = 15
        self.goal_poses_for_render = [
            [
                [-4, 12, 0],
                [-14, 4, 1.57079633],
            ],
            [
                [8, 10, 0],
                [12, 0, 0],
            ],
            [
                [0, -6, 3.14],
                [-8, -10, 3.14],
                [-2, -12, 1.57],
            ],
            [[8, -8, 0]],
        ]
        self.goal_poses = [
            [
                [5.5, 0.0, 13.5, 1.0, 0.0],
                [0.5, 0.0, 9.5, -3.2051033e-09, -1.0],
            ],
            [[11.5, 0.0, 12.5, 1.0, 0.0], [13.5, 0.0, 7.5, 1.0, 0.0]],
            [
                [7.5, 0.0, 4.5, -0.99999875, -0.0015926529],
                [3.5, 0.0, 2.5, -0.99999875, -0.0015926529],
                [6.5, 0.0, 1.5, 0.0007963267, -0.9999997],
            ],
            [[11.5, 0.0, 3.5, 1.0, 0.0]],
        ]
