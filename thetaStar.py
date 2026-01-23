"""
Theta* grid planning with:
  1) HARD obstacles (e.g., cans) the robot must never collide with
  2) SEMI-PERMEABLE boundaries the robot may move onto / along, but may NOT fully cross

Coordinate system: ORIGINAL (Cartesian)
  - x increases to the right
  - y increases upward
  - (0,0) is NOT forced to top-left

How semi-permeable boundaries work (key idea):
  - Boundary cells are traversable.
  - Space is partitioned into "regions" separated by boundary cells (computed once by flood-fill).
  - When the robot ENTERS a boundary from a region R, it may move within the boundary,
    but it may only EXIT back to region R (so it cannot cross to the other side).

This matches:
  - "can pass over it, but not fully cross"
  - can approach boundary and move close
  - cannot traverse through the boundary to the opposite side

Author: (original) Musab Kasbati (@Musab1Blaser)
Refactor: numpy grids + heap + semi-permeable boundaries + hard obstacles
Paper: https://cdn.aaai.org/AAAI/2007/AAAI07-187.pdf
"""

import math
import heapq
from collections import deque

import numpy as np
import matplotlib.pyplot as plt


show_animation = True
use_theta_star = True


class ThetaStarPlanner:
    def __init__(self, resolution: float, robot_radius: float):
        """
        Build planner object. You can add hard obstacles + boundaries, then call build_map().
        """
        self.resolution = float(resolution)
        self.rr = float(robot_radius)

        # World bounds (set in build_map)
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None

        # Grid size (set in build_map)
        self.x_width = None
        self.y_width = None

        # Maps (set in build_map)
        # obstacle_map: True = collision
        self.obstacle_map = None  # shape: (x_width, y_width) (matches original code indexing)
        # boundary_map: True = boundary cell (traversable)
        self.boundary_map = None  # shape: (x_width, y_width)
        # region_map: 0 = boundary/blocked/unlabeled, 1..K = free-space regions separated by boundaries
        self.region_map = None  # shape: (x_width, y_width), dtype int32

        # Inputs
        self._hard_obstacles = []   # list of (x, y) in world coords (e.g., can centers)
        self._hard_obstacle_rr = [] # per-obstacle radius inflation (world units); default robot rr
        self._boundaries = []       # list of polylines; each polyline is list of (x, y) world coords
        self._boundary_thickness = []  # thickness in world units

        # Motion model (dx, dy, cost) in GRID units
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
            dtype=float
        )

    # -------------------------
    # Public: add features
    # -------------------------

    def add_hard_obstacle_points(self, ox, oy, obstacle_radius=None):
        """
        Add HARD obstacles (e.g., cans) that the robot must never hit.

        ox, oy: lists (or arrays) of world x/y points.
        obstacle_radius: if None -> uses robot radius self.rr (inflation).
                        else -> uses given radius (world units) per point.
        """
        if obstacle_radius is None:
            obstacle_radius = self.rr

        if np.isscalar(obstacle_radius):
            for x, y in zip(ox, oy):
                self._hard_obstacles.append((float(x), float(y)))
                self._hard_obstacle_rr.append(float(obstacle_radius))
        else:
            # per-point radii
            for x, y, r in zip(ox, oy, obstacle_radius):
                self._hard_obstacles.append((float(x), float(y)))
                self._hard_obstacle_rr.append(float(r))

    def add_boundary_polyline(self, points, thickness=None):
        """
        Add a SEMI-PERMEABLE boundary as a polyline in world coords.
        The boundary is traversable but cannot be fully crossed.

        points: [(x0,y0), (x1,y1), ...]
        thickness: world units; if None -> thickness = resolution (about 1 cell thick)
        """
        if thickness is None:
            thickness = self.resolution

        pts = [(float(x), float(y)) for x, y in points]
        if len(pts) < 2:
            raise ValueError("Boundary polyline must have at least 2 points.")

        self._boundaries.append(pts)
        self._boundary_thickness.append(float(thickness))

    # -------------------------
    # Build maps
    # -------------------------

    def build_map(self, sx, sy, gx, gy, margin_cells=10):
        """
        Build obstacle, boundary, and region maps.
        Must be called after adding obstacles/boundaries and before planning().

        sx,sy,gx,gy are world coords (to include in bounds).
        margin_cells adds padding to ensure enough space around everything.
        """
        # 1) Determine bounds from start/goal + features
        xs = [float(sx), float(gx)]
        ys = [float(sy), float(gy)]

        if self._hard_obstacles:
            xs.extend([p[0] for p in self._hard_obstacles])
            ys.extend([p[1] for p in self._hard_obstacles])

        for poly in self._boundaries:
            xs.extend([p[0] for p in poly])
            ys.extend([p[1] for p in poly])

        # Expand bounds by margin (in world units)
        margin = margin_cells * self.resolution
        self.min_x = math.floor(min(xs) - margin)
        self.min_y = math.floor(min(ys) - margin)
        self.max_x = math.ceil(max(xs) + margin)
        self.max_y = math.ceil(max(ys) + margin)

        self.x_width = int(round((self.max_x - self.min_x) / self.resolution))
        self.y_width = int(round((self.max_y - self.min_y) / self.resolution))

        if self.x_width <= 0 or self.y_width <= 0:
            raise ValueError("Invalid grid size computed. Check bounds/resolution.")

        # 2) Allocate maps (keep original indexing style: [x][y])
        self.obstacle_map = np.zeros((self.x_width, self.y_width), dtype=bool)
        self.boundary_map = np.zeros((self.x_width, self.y_width), dtype=bool)
        self.region_map = np.zeros((self.x_width, self.y_width), dtype=np.int32)

        # 3) Rasterize hard obstacles (fast stamping with disk mask per unique radius)
        self._rasterize_hard_obstacles()

        # 4) Rasterize boundaries (as traversable boundary cells)
        self._rasterize_boundaries()

        # 5) Compute region labels (free space separated by boundary cells)
        self._compute_regions()

        # Optional debug prints
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)
        print("regions:", int(self.region_map.max()))

    def _rasterize_hard_obstacles(self):
        if not self._hard_obstacles:
            return

        # Group points by radius to reuse disk masks
        by_r = {}
        for (x, y), r in zip(self._hard_obstacles, self._hard_obstacle_rr):
            by_r.setdefault(r, []).append((x, y))

        for r_world, pts in by_r.items():
            r_cells = int(math.ceil(r_world / self.resolution))
            if r_cells <= 0:
                # mark nearest cell only
                for xw, yw in pts:
                    ix = self.calc_xy_index(xw, self.min_x)
                    iy = self.calc_xy_index(yw, self.min_y)
                    if 0 <= ix < self.x_width and 0 <= iy < self.y_width:
                        self.obstacle_map[ix, iy] = True
                continue

            # Disk offsets in cells
            rr2 = r_cells * r_cells
            dx = np.arange(-r_cells, r_cells + 1, dtype=np.int32)
            dy = np.arange(-r_cells, r_cells + 1, dtype=np.int32)
            DX, DY = np.meshgrid(dx, dy, indexing="ij")
            mask = (DX * DX + DY * DY) <= rr2
            off_x = DX[mask].ravel()
            off_y = DY[mask].ravel()

            for xw, yw in pts:
                cx = self.calc_xy_index(xw, self.min_x)
                cy = self.calc_xy_index(yw, self.min_y)

                xs = cx + off_x
                ys = cy + off_y
                valid = (
                    (xs >= 0) & (xs < self.x_width) &
                    (ys >= 0) & (ys < self.y_width)
                )
                self.obstacle_map[xs[valid], ys[valid]] = True

    def _rasterize_boundaries(self):
        if not self._boundaries:
            return

        for poly, thick_world in zip(self._boundaries, self._boundary_thickness):
            thick_cells = max(0, int(math.ceil(thick_world / self.resolution)))
            # Rasterize each segment
            for (x0, y0), (x1, y1) in zip(poly[:-1], poly[1:]):
                ix0 = self.calc_xy_index(x0, self.min_x)
                iy0 = self.calc_xy_index(y0, self.min_y)
                ix1 = self.calc_xy_index(x1, self.min_x)
                iy1 = self.calc_xy_index(y1, self.min_y)

                cells = self._bresenham_cells(ix0, iy0, ix1, iy1)
                for cx, cy in cells:
                    if 0 <= cx < self.x_width and 0 <= cy < self.y_width:
                        self.boundary_map[cx, cy] = True

            # Thicken boundary by dilation (still traversable)
            if thick_cells > 0:
                self._dilate_boundary(thick_cells)

        # Important: boundary is NOT an obstacle. Ensure obstacle_map doesn't override boundary unless you want.
        # If a hard obstacle overlaps boundary, it should remain hard obstacle:
        # (obstacle_map wins in collision checking)

    def _dilate_boundary(self, r_cells: int):
        # Stamp a disk around all boundary cells (simple and robust)
        boundary_points = np.argwhere(self.boundary_map)  # returns (x, y) pairs
        if boundary_points.size == 0:
            return

        rr2 = r_cells * r_cells
        dx = np.arange(-r_cells, r_cells + 1, dtype=np.int32)
        dy = np.arange(-r_cells, r_cells + 1, dtype=np.int32)
        DX, DY = np.meshgrid(dx, dy, indexing="ij")
        mask = (DX * DX + DY * DY) <= rr2
        off_x = DX[mask].ravel()
        off_y = DY[mask].ravel()

        for cx, cy in boundary_points:
            xs = cx + off_x
            ys = cy + off_y
            valid = (
                (xs >= 0) & (xs < self.x_width) &
                (ys >= 0) & (ys < self.y_width)
            )
            self.boundary_map[xs[valid], ys[valid]] = True

    def _compute_regions(self):
        """
        Label connected free-space regions separated by boundary cells.
        - Obstacles are blocked.
        - Boundary cells are NOT part of any region (label 0).
        """
        region_id = 0
        visited = np.zeros((self.x_width, self.y_width), dtype=bool)

        def is_free(ix, iy):
            if ix < 0 or ix >= self.x_width or iy < 0 or iy >= self.y_width:
                return False
            if self.obstacle_map[ix, iy]:
                return False
            if self.boundary_map[ix, iy]:
                return False
            return True

        for ix in range(self.x_width):
            for iy in range(self.y_width):
                if visited[ix, iy]:
                    continue
                if not is_free(ix, iy):
                    visited[ix, iy] = True
                    continue

                region_id += 1
                # BFS flood fill
                q = deque()
                q.append((ix, iy))
                visited[ix, iy] = True
                self.region_map[ix, iy] = region_id

                while q:
                    x, y = q.popleft()
                    # 4-connected is enough to define regions (boundaries are thin)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.x_width and 0 <= ny < self.y_width and not visited[nx, ny]:
                            visited[nx, ny] = True
                            if is_free(nx, ny):
                                self.region_map[nx, ny] = region_id
                                q.append((nx, ny))

    # -------------------------
    # Planning
    # -------------------------

    def planning(self, sx, sy, gx, gy):
        """
        Theta* search with semi-permeable boundaries.
        Returns rx, ry in world coords.
        """
        if self.obstacle_map is None:
            raise RuntimeError("Call build_map(...) before planning().")

        sx_i = self.calc_xy_index(sx, self.min_x)
        sy_i = self.calc_xy_index(sy, self.min_y)
        gx_i = self.calc_xy_index(gx, self.min_x)
        gy_i = self.calc_xy_index(gy, self.min_y)

        if not self._valid_cell(sx_i, sy_i):
            raise ValueError("Start is invalid or in collision.")
        if not self._valid_cell(gx_i, gy_i):
            raise ValueError("Goal is invalid or in collision.")

        start_is_boundary = self.boundary_map[sx_i, sy_i]
        goal_is_boundary = self.boundary_map[gx_i, gy_i]
        if start_is_boundary or goal_is_boundary:
            raise ValueError("Start/goal must not be on a boundary cell for semi-permeable logic to be well-defined.")

        start_region = int(self.region_map[sx_i, sy_i])
        goal_region = int(self.region_map[gx_i, gy_i])
        if start_region == 0 or goal_region == 0:
            raise ValueError("Start/goal must be in free space (not boundary / not blocked).")

        # Node state:
        #   (x, y, home_region)
        # home_region = -1 for normal free space
        # home_region = region_id when you are on boundary, meaning "must exit back to this region"
        start_state = (sx_i, sy_i, -1)
        goal_state = (gx_i, gy_i, -1)

        def h(x, y):
            return math.hypot(gx_i - x, gy_i - y)

        # g cost store
        g_cost = {start_state: 0.0}
        parent = {start_state: None}  # parent[state] = prev_state

        # Heap: (f, g, state)
        pq = [(h(sx_i, sy_i), 0.0, start_state)]
        closed = set()

        expand_count = 0

        while pq:
            _, gcur, cur = heapq.heappop(pq)
            if cur in closed:
                continue
            closed.add(cur)

            x, y, home = cur
            expand_count += 1

            # Bring back throttled pause (like original)
            if show_animation and (expand_count % 10 == 0):
                plt.pause(0.00001)

            if (x, y) == (gx_i, gy_i) and home == -1:
                # Found goal in free space
                break

            # Expand neighbors
            for dx, dy, step_cost in self.motion:
                nx = x + int(dx)
                ny = y + int(dy)

                # Must stay in grid
                if nx < 0 or nx >= self.x_width or ny < 0 or ny >= self.y_width:
                    continue

                # Hard obstacles are never allowed
                if self.obstacle_map[nx, ny]:
                    continue

                n_is_boundary = bool(self.boundary_map[nx, ny])
                c_is_boundary = bool(self.boundary_map[x, y])

                # Semi-permeable rules:
                # - If moving into boundary from free space: set home_region = current region
                # - If moving within boundary: keep home_region
                # - If moving from boundary to free space: only allowed if exiting back to home_region
                if not c_is_boundary:
                    c_region = int(self.region_map[x, y])
                    if c_region == 0:
                        continue  # should not happen unless map issue

                    if n_is_boundary:
                        n_home = c_region
                    else:
                        n_region = int(self.region_map[nx, ny])
                        if n_region == 0:
                            continue  # can't step into boundary-labeled or blocked (should be boundary)
                        # You cannot step into a different free region without going through boundary.
                        # This prevents "crossing" in free space directly.
                        if n_region != c_region:
                            continue
                        n_home = -1
                else:
                    # current is boundary
                    if home <= 0:
                        # boundary states must have valid home_region
                        continue
                    if n_is_boundary:
                        n_home = home
                    else:
                        n_region = int(self.region_map[nx, ny])
                        if n_region != home:
                            # Can't exit boundary into the other side
                            continue
                        n_home = -1

                n_state = (nx, ny, n_home)
                if n_state in closed:
                    continue

                # Base cost for step
                tentative_g = gcur + float(step_cost)

                # Theta* shortcut only applies within the same "mode"
                # (we avoid compressing across boundary logic; keep it correct and simple)
                if use_theta_star and n_home == -1 and home == -1:
                    # Try connect from parent of current (if any) to neighbor
                    pstate = parent.get(cur)
                    if pstate is not None:
                        px, py, phome = pstate
                        if phome == -1:
                            # Ensure same region and line-of-sight
                            if self.region_map[px, py] == self.region_map[nx, ny] and self.line_of_sight(px, py, nx, ny):
                                pg = g_cost[pstate]
                                los_g = pg + math.hypot(nx - px, ny - py)
                                if los_g < tentative_g:
                                    tentative_g = los_g
                                    # change parent for n_state
                                    # (note: parent state is free-space, and n_state is free-space)
                                    parent[n_state] = pstate

                # Normal relaxation
                if tentative_g < g_cost.get(n_state, float("inf")):
                    g_cost[n_state] = tentative_g
                    if n_state not in parent:
                        parent[n_state] = cur
                    else:
                        # only overwrite parent if we didn't already set it via theta* above
                        if parent[n_state] != parent.get(n_state):
                            parent[n_state] = cur

                    f = tentative_g + h(nx, ny)
                    heapq.heappush(pq, (f, tentative_g, n_state))

        # Reconstruct
        if goal_state not in parent and goal_state not in g_cost:
            # might be unreachable
            return [], []

        rx, ry = self._reconstruct_path(parent, goal_state)
        return rx, ry

    def _reconstruct_path(self, parent, goal_state):
        cur = goal_state
        if cur not in parent:
            return [], []

        rx, ry = [], []
        while cur is not None:
            x, y, _home = cur
            rx.append(self.calc_grid_position(x, self.min_x))
            ry.append(self.calc_grid_position(y, self.min_y))
            cur = parent[cur]

        rx.reverse()
        ry.reverse()
        return rx, ry

    # -------------------------
    # Geometry + grid helpers
    # -------------------------

    def calc_grid_position(self, index, min_position):
        return index * self.resolution + min_position

    def calc_xy_index(self, position, min_pos):
        return int(round((position - min_pos) / self.resolution))

    def _valid_cell(self, x, y):
        if x < 0 or x >= self.x_width or y < 0 or y >= self.y_width:
            return False
        if self.obstacle_map[x, y]:
            return False
        return True

    def line_of_sight(self, x0, y0, x1, y1):
        """
        Bresenham LOS on grid cells; treats hard obstacles as blocking.
        Boundary cells do NOT block LOS (they are traversable).
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            if x0 < 0 or x0 >= self.x_width or y0 < 0 or y0 >= self.y_width:
                return False
            if self.obstacle_map[x0, y0]:
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

    @staticmethod
    def _bresenham_cells(x0, y0, x1, y1):
        """
        Return list of grid cells (x,y) along segment using Bresenham.
        """
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells


def main():
    print(__file__ + " start!!")

    # start and goal position (world coords, original Cartesian)
    sx, sy = 10.0, 10.0
    gx, gy = 50.0, 50.0

    grid_size = 0.5      # [m]
    robot_radius = 1.0   # [m]

    planner = ThetaStarPlanner(resolution=grid_size, robot_radius=robot_radius)

    # -------------------------
    # Example: add HARD obstacles (cans)
    # -------------------------
    # These points are "never hit" obstacles. Increase obstacle_radius to keep extra distance from cans.
    can_x = [30.0, 31.0, 32.0]
    can_y = [30.0, 30.5, 29.7]
    planner.add_hard_obstacle_points(can_x, can_y, obstacle_radius=1.2)  # slightly > robot_radius

    # -------------------------
    # Example: add SEMI-PERMEABLE boundaries
    # -------------------------
    # A "fence" line the robot can step onto but cannot fully cross.
    fence = [(20.0, -10.0), (20.0, 60.0)]
    planner.add_boundary_polyline(fence, thickness=0.5)

    # Another boundary (diagonal)
    fence2 = [(40.0, 60.0), (60.0, 40.0)]
    planner.add_boundary_polyline(fence2, thickness=0.5)

    # -------------------------
    # Also add the outer walls as HARD obstacles (classic closed box)
    # (If you want them to be semi-permeable instead, add them as boundaries, not obstacles.)
    # -------------------------
    ox, oy = [], []
    for i in range(-10, 61):
        ox.append(i); oy.append(-10.0)
        ox.append(i); oy.append(60.0)
    for i in range(-10, 61):
        ox.append(-10.0); oy.append(i)
        ox.append(60.0);  oy.append(i)
    planner.add_hard_obstacle_points(ox, oy, obstacle_radius=robot_radius)

    # Build maps (bounds computed from features + start/goal)
    planner.build_map(sx, sy, gx, gy, margin_cells=10)

    # -------------------------
    # Visualization
    # -------------------------
    if show_animation:
        plt.figure()
        # draw hard obstacles
        obs = np.argwhere(planner.obstacle_map)  # (x,y)
        if obs.size:
            wx = [planner.calc_grid_position(int(x), planner.min_x) for x, y in obs]
            wy = [planner.calc_grid_position(int(y), planner.min_y) for x, y in obs]
            plt.plot(wx, wy, ".k", markersize=2)

        # draw boundaries
        bnd = np.argwhere(planner.boundary_map)
        if bnd.size:
            bx = [planner.calc_grid_position(int(x), planner.min_x) for x, y in bnd]
            by = [planner.calc_grid_position(int(y), planner.min_y) for x, y in bnd]
            plt.plot(bx, by, ".y", markersize=2)

        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        plt.grid(True)
        plt.axis("equal")

    # Plan
    rx, ry = planner.planning(sx, sy, gx, gy)

    print("rx:", rx)
    print("ry:", ry)

    if show_animation:
        if rx and ry:
            plt.plot(rx, ry, "-r", linewidth=2)
        plt.pause(0.01)
        plt.show()


if __name__ == "__main__":
    main()
