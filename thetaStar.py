"""
Theta* grid planning (fast NumPy + heapq version)

author: Musab Kasbati (@Musab1Blaser)  (original)
refactor: faster open-set, NumPy grids, faster obstacle rasterization

See paper: https://cdn.aaai.org/AAAI/2007/AAAI07-187.pdf
Also: https://atsushisakai.github.io/PythonRobotics/modules/5_path_planning/grid_base_search/grid_base_search.html
"""

import math
import heapq
import numpy as np
import matplotlib.pyplot as plt

show_animation = True
use_theta_star = True


class ThetaStarPlanner:
    def __init__(self, ox, oy, resolution, rr):
        """
        ox, oy: obstacle world coords (lists)
        resolution: grid resolution [m]
        rr: robot radius [m]
        """
        self.resolution = float(resolution)
        self.rr = float(rr)

        self.min_x = int(round(min(ox)))
        self.min_y = int(round(min(oy)))
        self.max_x = int(round(max(ox)))
        self.max_y = int(round(max(oy)))

        self.x_width = int(round((self.max_x - self.min_x) / self.resolution))
        self.y_width = int(round((self.max_y - self.min_y) / self.resolution))

        # 8-connected motion: dx, dy, cost
        self.motion = np.array(
            [
                [1, 0, 1.0],
                [0, 1, 1.0],
                [-1, 0, 1.0],
                [0, -1, 1.0],
                [-1, -1, math.sqrt(2)],
                [-1, 1, math.sqrt(2)],
                [1, -1, math.sqrt(2)],
                [1, 1, math.sqrt(2)],
            ],
            dtype=float,
        )

        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

        self._build_obstacle_map_fast(ox, oy)

    # -------------------------
    # Public API
    # -------------------------

    def planning(self, sx, sy, gx, gy):
        """
        Theta* / A* planning on a grid.
        Returns:
            rx, ry: lists of world coordinates for the path
        """
        sx_i = self._xy_to_index(sx, self.min_x)
        sy_i = self._xy_to_index(sy, self.min_y)
        gx_i = self._xy_to_index(gx, self.min_x)
        gy_i = self._xy_to_index(gy, self.min_y)

        if not self._valid(sx_i, sy_i):
            raise ValueError("Start is invalid or in collision.")
        if not self._valid(gx_i, gy_i):
            raise ValueError("Goal is invalid or in collision.")

        INF = np.inf

        g_cost = np.full((self.x_width, self.y_width), INF, dtype=float)
        closed = np.zeros((self.x_width, self.y_width), dtype=bool)

        # parent[x,y] = (px, py)
        parent = np.full((self.x_width, self.y_width, 2), -1, dtype=np.int32)

        g_cost[sx_i, sy_i] = 0.0
        parent[sx_i, sy_i] = (sx_i, sy_i)

        # (f, x, y)
        pq = []
        heapq.heappush(pq, (self._heuristic(sx_i, sy_i, gx_i, gy_i), sx_i, sy_i))

        expand_count = 0

        # Optional live plot setup
        if show_animation:
            # draw explored nodes as small dots (cheap)
            explored_x = []
            explored_y = []

        while pq:
            _, x, y = heapq.heappop(pq)
            if closed[x, y]:
                continue
            closed[x, y] = True
            expand_count += 1

            # ---- bring back the original throttled pause ----
            # (equivalent to: if len(closed_set.keys()) % 10 == 0: plt.pause(0.00001))
            if show_animation and (expand_count % 10 == 0):
                # add a tiny bit of drawing so pause actually updates something
                explored_x.append(self._index_to_xy(x, self.min_x))
                explored_y.append(self._index_to_xy(y, self.min_y))
                plt.plot(explored_x[-1], explored_y[-1], ".c", markersize=2)
                plt.pause(0.00001)

            if x == gx_i and y == gy_i:
                break

            px, py = parent[x, y]

            for dx, dy, step_cost in self.motion:
                nx = x + int(dx)
                ny = y + int(dy)

                if not self._valid(nx, ny) or closed[nx, ny]:
                    continue

                # default: A* relaxation from current
                best_parent_x, best_parent_y = x, y
                new_g = g_cost[x, y] + float(step_cost)

                # Theta*: try connecting via parent(current) if line of sight exists
                if use_theta_star and self._line_of_sight(px, py, nx, ny):
                    los_g = g_cost[px, py] + math.hypot(nx - px, ny - py)
                    if los_g < new_g:
                        new_g = los_g
                        best_parent_x, best_parent_y = px, py

                if new_g < g_cost[nx, ny]:
                    g_cost[nx, ny] = new_g
                    parent[nx, ny] = (best_parent_x, best_parent_y)
                    f = new_g + self._heuristic(nx, ny, gx_i, gy_i)
                    heapq.heappush(pq, (f, nx, ny))

        rx, ry = self._reconstruct_path(parent, sx_i, sy_i, gx_i, gy_i)
        return rx, ry

    # -------------------------
    # Path reconstruction
    # -------------------------

    def _reconstruct_path(self, parent, sx, sy, gx, gy):
        x, y = gx, gy
        rx = [self._index_to_xy(x, self.min_x)]
        ry = [self._index_to_xy(y, self.min_y)]

        # If goal is unreachable, parent will be (-1,-1)
        if parent[x, y, 0] < 0:
            return [], []

        while not (x == sx and y == sy):
            px, py = parent[x, y]
            # safety against malformed parents
            if px < 0:
                return [], []
            x, y = int(px), int(py)
            rx.append(self._index_to_xy(x, self.min_x))
            ry.append(self._index_to_xy(y, self.min_y))

        rx.reverse()
        ry.reverse()
        return rx, ry

    # -------------------------
    # Grid / geometry helpers
    # -------------------------

    @staticmethod
    def _heuristic(x, y, gx, gy):
        # Euclidean heuristic (same as original)
        return math.hypot(gx - x, gy - y)

    def _xy_to_index(self, pos, min_pos):
        return int(round((pos - min_pos) / self.resolution))

    def _index_to_xy(self, index, min_pos):
        return index * self.resolution + min_pos

    def _valid(self, x, y):
        return (
            0 <= x < self.x_width
            and 0 <= y < self.y_width
            and (not self.obstacle_map[x, y])
        )

    def _line_of_sight(self, x0, y0, x1, y1):
        """
        Bresenham line traversal in grid coordinates, checking collision along the segment.
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if not self._valid(x0, y0):
                return False
            if x0 == x1 and y0 == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

    # -------------------------
    # Fast obstacle map
    # -------------------------

    def _build_obstacle_map_fast(self, ox, oy):
        """
        Faster obstacle map construction:
        Rasterize obstacles onto grid using a precomputed disk mask of radius rr.
        """
        self.obstacle_map = np.zeros((self.x_width, self.y_width), dtype=bool)

        ox = np.asarray(ox, dtype=float)
        oy = np.asarray(oy, dtype=float)

        # obstacle indices in grid
        gx = np.rint((ox - self.min_x) / self.resolution).astype(np.int32)
        gy = np.rint((oy - self.min_y) / self.resolution).astype(np.int32)

        # radius in cells
        r_cells = int(math.ceil(self.rr / self.resolution))
        if r_cells <= 0:
            # just mark nearest cells
            valid = (gx >= 0) & (gx < self.x_width) & (gy >= 0) & (gy < self.y_width)
            self.obstacle_map[gx[valid], gy[valid]] = True
            return

        # precompute disk mask offsets (dx, dy) satisfying dx^2 + dy^2 <= r_cells^2
        rr2 = r_cells * r_cells
        dx = np.arange(-r_cells, r_cells + 1, dtype=np.int32)
        dy = np.arange(-r_cells, r_cells + 1, dtype=np.int32)
        DX, DY = np.meshgrid(dx, dy, indexing="ij")
        mask = (DX * DX + DY * DY) <= rr2
        off_x = DX[mask].ravel()
        off_y = DY[mask].ravel()

        # stamp disk for each obstacle cell
        for cx, cy in zip(gx, gy):
            x_idx = cx + off_x
            y_idx = cy + off_y

            valid = (
                (x_idx >= 0)
                & (x_idx < self.x_width)
                & (y_idx >= 0)
                & (y_idx < self.y_width)
            )
            self.obstacle_map[x_idx[valid], y_idx[valid]] = True


ox, oy = [], []
# Max boundaries so theta star works
for i in range(-100, 100):
    ox.append(i)
    oy.append(-25)
for i in range(-25, 125):
    ox.append(100.0)
    oy.append(i)
for i in range(-100, 100):
    ox.append(i)
    oy.append(125.0)
for i in range(-25, 125):
    ox.append(-100.0)
    oy.append(i)
sx = 0.0  # [m]
sy = 3.0  # [m]
grid_size = 0.5  # [m]
robot_radius = 5.0  # [m]

left_cans_x, left_cans_y = -45, 45
right_cans_x, right_cans_y = 45, 45
red_zone_x, red_zone_y = 0, 0
green_zone_x, green_zone_y = 0, 0
yellow_zone_x, yellow_zone_y = 0, 0

def main():
    print(__file__ + " start!!")

    # start and goal position
    gx = 40.0  # [m]
    gy = 40.0  # [m]

    ox.append(20.0)
    oy.append(10)

    ox.append(15.0)
    oy.append(15)
    ox.append(15.0)
    oy.append(25)

    if show_animation:
        plt.figure()
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    theta_star = ThetaStarPlanner(ox, oy, grid_size, robot_radius)
    rx, ry = theta_star.planning(sx, sy, gx, gy)

    print(rx)
    print(ry)

    if show_animation:
        if rx and ry:
            plt.plot(rx, ry, "-r", linewidth=2)
        plt.pause(0.01)
        plt.show()


if __name__ == "__main__":
    main()
