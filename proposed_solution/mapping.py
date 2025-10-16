# mapping.py
import math
import numpy as np
from config import (
    MAP_RES_M, MAP_SIZE_M, LOG_FREE, LOG_OCC, LOG_MIN, LOG_MAX,
    HIT_EPS, SCAN_SUBSAMPLE_EVERY
)
from utils import bresenham, world_to_map

class OccupancyGrid:
    def __init__(self, map_res=MAP_RES_M, map_size=MAP_SIZE_M):
        self.map_res = map_res
        self.map_size = map_size
        self.size_cells = int(self.map_size / self.map_res)
        self.map_log = np.zeros((self.size_cells, self.size_cells), dtype=np.float32)
        self.map_origin_xy = (0.0, 0.0)
        self._need_set_origin = True

    def ensure_origin(self, x, y):
        if self._need_set_origin:
            self.map_origin_xy = (x, y)
            self._need_set_origin = False

    def update_from_lidar(self, get_pose, lidar):
        self.ensure_origin(*get_pose()[:2])
        x_r, y_r, yaw = get_pose()
        ranges = lidar.getRangeImage()
        res = lidar.getHorizontalResolution()
        layers = lidar.getNumberOfLayers()
        if layers > 1:
            ranges = ranges[0:res]
        fov = lidar.getFov()
        rmin, rmax = lidar.getMinRange(), lidar.getMaxRange()
        dtheta = fov / (res - 1) if res > 1 else 0.0

        ix0, iy0 = world_to_map(x_r, y_r, self.map_origin_xy, self.map_res, self.size_cells)

        for i in range(0, res, SCAN_SUBSAMPLE_EVERY):
            r = ranges[i]
            if np.isnan(r) or r <= rmin or r > rmax:
                continue
            theta = -fov/2.0 + i * dtheta
            beam_world = yaw + theta

            hit_dist = min(r + HIT_EPS, rmax)
            x_hit = x_r + hit_dist * math.sin(beam_world)
            y_hit = y_r + hit_dist * math.cos(beam_world)
            ix1, iy1 = world_to_map(x_hit, y_hit, self.map_origin_xy, self.map_res, self.size_cells)

            PROTECT_CELLS = 2
            line = bresenham(ix0, iy0, ix1, iy1)
            protect = set(line[-(PROTECT_CELLS + 1):-1]) if len(line) > PROTECT_CELLS else set()

            for (ix, iy) in line[:-1]:
                if 0 <= ix < self.size_cells and 0 <= iy < self.size_cells:
                    if (ix, iy) in protect:
                        continue
                    if self.map_log[iy, ix] >= 0.8:
                        continue
                    self.map_log[iy, ix] = np.clip(self.map_log[iy, ix] + LOG_FREE, LOG_MIN, LOG_MAX)

            if 0 <= ix1 < self.size_cells and 0 <= iy1 < self.size_cells:
                self.map_log[iy1, ix1] = np.clip(self.map_log[iy1, ix1] + LOG_OCC, LOG_MIN, LOG_MAX)

    def map_to_world(self, ix: int, iy: int):
        ox, oy = self.map_origin_xy
        x = (ix + 0.5) * self.map_res + (ox - self.size_cells * self.map_res / 2)
        y = (iy + 0.5) * self.map_res + (oy - self.size_cells * self.map_res / 2)
        return x, y

    # ---- nav helpers ----
    def grid_is_occupied(self, ix, iy, occ_th=0.7) -> bool:
        if not (0 <= ix < self.size_cells and 0 <= iy < self.size_cells):
            return True
        return self.map_log[iy, ix] >= occ_th

    def grid_cost(self, ix, iy, occ_th=0.7) -> float:
        if self.grid_is_occupied(ix, iy, occ_th):
            return float('inf')
        log = self.map_log[iy, ix]
        return 1.0 if log <= -0.7 else 1.5

    def inflated_is_blocked(self, ix, iy, occ_th=0.7, inflate_m=0.20) -> bool:
        r = max(1, int(round(inflate_m / self.map_res)))
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < self.size_cells and 0 <= jy < self.size_cells:
                    if self.map_log[jy, jx] >= occ_th:
                        return True
        return False

    def has_line_of_sight(self, a, b, occ_th=0.7, inflate_m=0.20) -> bool:
        for (ix, iy) in bresenham(a[0], a[1], b[0], b[1]):
            if self.inflated_is_blocked(ix, iy, occ_th, inflate_m):
                return False
        return True