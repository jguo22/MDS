"""
Theta* grid planning (grid-cells version + greedy LOS simplification)

- Planner operates purely in GRID CELLS (integers).
- Robot/world (mm) conversion handled by convert_world_to_grid / convert_grid_to_world.
- Obstacle map is built in grid-cells and inflated by robot_radius (in cells).

This avoids mixing "world coords" and "grid coords" which breaks LOS smoothing.
"""

import math
import heapq
import numpy as np
import matplotlib.pyplot as plt
from config import ROBOT_DIAMETER

show_animation = False
use_theta_star = True


# -------------------------
# Field + conversion (robot world mm <-> planner grid cells)
# -------------------------

FIELD_SIZE_MM = 10 * 304.8          # 3048 mm
GRID_SIZE = 100                      # 100 x 100 cells
CELL_SIZE_MM = FIELD_SIZE_MM / GRID_SIZE  # 30.48 mm per cell

GRID_W = 200
GRID_H = 200

LOGICAL_OFFSET = 50   # how much the 100x100 is inset
START_GRID_X = LOGICAL_OFFSET + GRID_SIZE // 2  # 50 + 50 = 100
START_GRID_Y = LOGICAL_OFFSET                   # 50
# START_GRID_X = 50
# START_GRID_Y = 0


def convert_world_to_grid(wx_mm: float, wy_mm: float):
    """
    Robot/world (mm) -> planner grid cells (int)

    Robot frame:
      +X forward
      +Y left

    Planner grid:
      +x right
      +y up

    Bottom-middle is (50,0).
    """
    gx_offset = -wy_mm / CELL_SIZE_MM
    gy_offset = wx_mm / CELL_SIZE_MM

    gx = int(round(START_GRID_X + gx_offset))
    gy = int(round(START_GRID_Y + gy_offset))
    return gx, gy


def convert_grid_to_world(gx: int, gy: int):
    """
    Planner grid cells -> robot/world (mm)
    """
    dx = gx - START_GRID_X
    dy = gy - START_GRID_Y

    wx_mm = dy * CELL_SIZE_MM
    wy_mm = -dx * CELL_SIZE_MM
    return wx_mm, wy_mm


def clamp_grid(gx: int, gy: int):
    gx = max(0, min(GRID_W - 1, gx))
    gy = max(0, min(GRID_H - 1, gy))
    return gx, gy


# -------------------------
# Theta* Planner (GRID CELLS ONLY)
# -------------------------

class ThetaStarPlanner:
    def __init__(self, ox, oy, grid_w, grid_h, robot_radius_cells: float):
        """
        ox, oy: obstacle coordinates in GRID CELLS
        grid_w, grid_h: grid dimensions
        robot_radius_cells: inflation radius in CELLS (keep your same value)
        """
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.rr = float(robot_radius_cells)

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

        self._build_obstacle_map_fast(ox, oy)

    def planning(self, sx, sy, gx, gy):
        """
        Theta* / A* planning on a grid (ALL IN GRID CELLS).
        Returns:
            rx, ry: lists of grid-cell coordinates
        """
        sx, sy = int(sx), int(sy)
        gx, gy = int(gx), int(gy)

        if not self._valid(sx, sy):
            raise ValueError("Start is invalid or in collision.")
        if not self._valid(gx, gy):
            raise ValueError("Goal is invalid or in collision.")

        INF = np.inf
        g_cost = np.full((self.grid_w, self.grid_h), INF, dtype=float)
        closed = np.zeros((self.grid_w, self.grid_h), dtype=bool)
        parent = np.full((self.grid_w, self.grid_h, 2), -1, dtype=np.int32)

        g_cost[sx, sy] = 0.0
        parent[sx, sy] = (sx, sy)

        pq = []
        heapq.heappush(pq, (self._heuristic(sx, sy, gx, gy), sx, sy))

        expand_count = 0

        while pq:
            _, x, y = heapq.heappop(pq)
            if closed[x, y]:
                continue
            closed[x, y] = True
            expand_count += 1

            if show_animation and (expand_count % 10 == 0):
                plt.plot(x, y, ".c", markersize=2)
                plt.pause(0.00001)

            if x == gx and y == gy:
                break

            px, py = parent[x, y]

            for dx, dy, step_cost in self.motion:
                nx = x + int(dx)
                ny = y + int(dy)

                if not self._valid(nx, ny) or closed[nx, ny]:
                    continue

                # Default A*
                best_parent_x, best_parent_y = x, y
                new_g = g_cost[x, y] + float(step_cost)

                # Theta*: try connect via parent(current) if LOS exists
                if use_theta_star and self._line_of_sight(px, py, nx, ny):
                    los_g = g_cost[px, py] + math.hypot(nx - px, ny - py)
                    if los_g < new_g:
                        new_g = los_g
                        best_parent_x, best_parent_y = px, py

                if new_g < g_cost[nx, ny]:
                    g_cost[nx, ny] = new_g
                    parent[nx, ny] = (best_parent_x, best_parent_y)
                    f = new_g + self._heuristic(nx, ny, gx, gy)
                    heapq.heappush(pq, (f, nx, ny))

        rx, ry = self._reconstruct_path(parent, sx, sy, gx, gy)
        return rx, ry

    @staticmethod
    def _heuristic(x, y, gx, gy):
        return math.hypot(gx - x, gy - y)

    def _valid(self, x, y):
        return (
            0 <= x < self.grid_w
            and 0 <= y < self.grid_h
            and (not self.obstacle_map[x, y])
        )

    def _reconstruct_path(self, parent, sx, sy, gx, gy):
        x, y = gx, gy
        if parent[x, y, 0] < 0:
            return [], []

        rx = [x]
        ry = [y]

        while not (x == sx and y == sy):
            px, py = parent[x, y]
            if px < 0:
                return [], []
            x, y = int(px), int(py)
            rx.append(x)
            ry.append(y)

        rx.reverse()
        ry.reverse()
        return rx, ry

    def _line_of_sight(self, x0, y0, x1, y1):
        """
        Bresenham line traversal in GRID CELLS, collision-checking along the segment.
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

    def _build_obstacle_map_fast(self, ox, oy):
        """
        Build obstacle map in GRID CELLS.
        Inflates obstacles by self.rr (cells).
        """
        self.obstacle_map = np.zeros((self.grid_w, self.grid_h), dtype=bool)

        ox = np.asarray(ox, dtype=np.int32)
        oy = np.asarray(oy, dtype=np.int32)

        # radius in cells (keep your same "robot_radius" concept)
        r_cells = int(math.ceil(self.rr))
        if r_cells <= 0:
            valid = (
                ox >= 0) & (
                ox < self.grid_w) & (
                oy >= 0) & (
                oy < self.grid_h)
            self.obstacle_map[ox[valid], oy[valid]] = True
            return

        rr2 = r_cells * r_cells
        dx = np.arange(-r_cells, r_cells + 1, dtype=np.int32)
        dy = np.arange(-r_cells, r_cells + 1, dtype=np.int32)
        DX, DY = np.meshgrid(dx, dy, indexing="ij")
        mask = (DX * DX + DY * DY) <= rr2
        off_x = DX[mask].ravel()
        off_y = DY[mask].ravel()

        for cx, cy in zip(ox, oy):
            x_idx = cx + off_x
            y_idx = cy + off_y
            valid = (
                (x_idx >= 0) & (x_idx < self.grid_w) &
                (y_idx >= 0) & (y_idx < self.grid_h)
            )
            self.obstacle_map[x_idx[valid], y_idx[valid]] = True

    # -------------------------
    # Path simplification (GRID CELLS)
    # -------------------------

    def simplify_path_los_greedy(self, rx, ry):
        """
        Greedy LOS shortcutting:
        Builds the longest visible segment repeatedly => fewest turns in practice.
        Input/Output in GRID CELLS.
        """
        if not rx or len(rx) < 3:
            return rx, ry

        pts = list(zip(rx, ry))
        out = [pts[0]]

        i = 0
        while i < len(pts) - 1:
            # extend j as far as LOS holds
            j = i + 1
            last_good = j

            while j < len(pts):
                x0, y0 = out[-1]
                x1, y1 = pts[j]
                if self._line_of_sight(x0, y0, x1, y1):
                    last_good = j
                    j += 1
                else:
                    break

            out.append(pts[last_good])
            i = last_good

        rx2, ry2 = zip(*out)
        return list(rx2), list(ry2)


# -------------------------
# Example main
# -------------------------


class ThetaStar():
    def __init__(self):
        self.border_ox, self.border_oy = [], []  # Static border obstacles
        self.dynamic_obstacles: dict[tuple[int, int], set[tuple[int, int]]] = {}  # Track added obstacles
        self.robot_radius = 1  # 177 mm / 30.48 mm = 5.8 cells
        self.obstacle_radius_cells = int(math.ceil((ROBOT_DIAMETER / 2) / CELL_SIZE_MM))

        # --- Start/goal from robot-world (mm) -> grid
        self.sx, self.sy = convert_world_to_grid(0, 0)
        self.gx, self.gy = convert_world_to_grid(0, 0)
        self.sx, self.sy = clamp_grid(self.sx, self.sy)
        self.gx, self.gy = clamp_grid(self.gx, self.gy)

        # Make borders
        MIN_B = 0
        MAX_B = GRID_W - 1   # 199

        # bottom & top
        for x in range(MIN_B, MAX_B + 1):
            self.border_ox.append(x)
            self.border_oy.append(MIN_B)     # y = -50 mapped to 0
            self.border_ox.append(x)
            self.border_oy.append(MAX_B)     # y = 150 mapped to 199

        # left & right
        for y in range(MIN_B, MAX_B + 1):
            self.border_ox.append(MIN_B)
            self.border_oy.append(y)
            self.border_ox.append(MAX_B)
            self.border_oy.append(y)

    @property
    def ox(self):
        """Combined list of all obstacle x coordinates."""
        all_ox = list(self.border_ox)
        for cells in self.dynamic_obstacles.values():
            for gx, _ in cells:
                all_ox.append(gx)
        return all_ox

    @property
    def oy(self):
        """Combined list of all obstacle y coordinates."""
        all_oy = list(self.border_oy)
        for cells in self.dynamic_obstacles.values():
            for _, gy in cells:
                all_oy.append(gy)
        return all_oy

    def set_start(self, world_x, world_y):
        self.sx, self.sy = convert_world_to_grid(world_x, world_y)
        self.sx, self.sy = clamp_grid(self.sx, self.sy)

    def set_goal(self, world_x, world_y):
        self.gx, self.gy = convert_world_to_grid(world_x, world_y)
        self.gx, self.gy = clamp_grid(self.gx, self.gy)

    def prune_near_collinear(self, rx, ry, angle_deg=10.0):
        """
        Remove points that introduce only small heading changes.
        Input/Output in GRID CELLS.
        """
        if len(rx) < 3:
            return rx, ry

        outx, outy = [rx[0]], [ry[0]]

        def angle(ax, ay, bx, by):
            da = math.hypot(ax, ay)
            db = math.hypot(bx, by)
            if da == 0 or db == 0:
                return 0.0
            c = max(-1.0, min(1.0, (ax * bx + ay * by) / (da * db)))
            return math.degrees(math.acos(c))

        for i in range(1, len(rx) - 1):
            v1x, v1y = rx[i] - rx[i - 1], ry[i] - ry[i - 1]
            v2x, v2y = rx[i + 1] - rx[i], ry[i + 1] - ry[i]
            if angle(v1x, v1y, v2x, v2y) > angle_deg:
                outx.append(rx[i])
                outy.append(ry[i])

        outx.append(rx[-1])
        outy.append(ry[-1])
        return outx, outy

    def addCan(self, world_x, world_y):
        """Add a can as an obstacle (uses robot_radius for inflation)."""
        self.addObstacle(world_x, world_y, radius_cells=self.robot_radius)

    def addObstacle(self, world_x, world_y, radius_cells: int | None = None):
        """
        Add an obstacle at world coordinates with circular inflation.

        Args:
            world_x: X position in mm
            world_y: Y position in mm
            radius_cells: Inflation radius in grid cells (default: ROBOT_DIAMETER/2 in cells)
        """
        if radius_cells is None:
            radius_cells = self.obstacle_radius_cells

        center_gx, center_gy = convert_world_to_grid(world_x, world_y)
        key = (center_gx, center_gy)

        if key in self.dynamic_obstacles:
            return  # Already added

        cells = set()
        for i in range(-radius_cells, radius_cells + 1):
            for j in range(-radius_cells, radius_cells + 1):
                if i * i + j * j <= radius_cells * radius_cells:
                    obs_gx, obs_gy = clamp_grid(center_gx + i, center_gy + j)
                    cells.add((obs_gx, obs_gy))

        self.dynamic_obstacles[key] = cells

    def removeObstacle(self, world_x, world_y):
        """
        Remove an obstacle at world coordinates.

        Args:
            world_x: X position in mm
            world_y: Y position in mm

        Returns:
            True if obstacle was removed, False if not found
        """
        center_gx, center_gy = convert_world_to_grid(world_x, world_y)
        key = (center_gx, center_gy)

        if key in self.dynamic_obstacles:
            del self.dynamic_obstacles[key]
            return True
        return False

    def clearObstacles(self):
        """Remove all dynamically added obstacles."""
        self.dynamic_obstacles.clear()


    def path_find(self):
        planner = ThetaStarPlanner(
            self.ox, self.oy, GRID_W, GRID_H, self.robot_radius)
        rx, ry = planner.planning(self.sx, self.sy, self.gx, self.gy)
        rx, ry = planner.simplify_path_los_greedy(rx, ry)
        rx, ry = self.prune_near_collinear(rx, ry, angle_deg=10.0)
        return [convert_grid_to_world(x, y) for x, y in zip(rx, ry)]


def main():
    thetaStar = ThetaStar()
    thetaStar.addCan(700, 0)
    thetaStar.set_goal(-2000, 0)

    # # --- Plot setup
    # if show_animation:
    #     plt.figure()
    #     plt.plot(thetaStar.ox, thetaStar.oy, ".k")
    #     plt.plot(thetaStar.sx, thetaStar.sy, "og")
    #     plt.plot(thetaStar.gx, thetaStar.gy, "xb")
    #     plt.grid(True)
    #     plt.axis("equal")
    #     plt.title("Theta* in GRID CELLS")

    # # --- Plot final
    # if show_animation and thetaStar.rx and ry:
    #     plt.plot(rx, ry, "-r", linewidth=2)
    #     plt.show()

    # --- If you want to send to robot: convert each waypoint to world mm
    world_waypoints = thetaStar.path_find()
    print("world waypoints (mm):")
    for w in world_waypoints:
        print(w)


if __name__ == "__main__":
    main()
