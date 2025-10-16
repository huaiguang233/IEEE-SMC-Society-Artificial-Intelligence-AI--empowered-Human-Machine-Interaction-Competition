# visualization.py
import cv2, math, numpy as np
from config import *

def render_map(grid, get_pose, robot_time, show_window=True, save_every_s=2.0):
    log = grid.map_log
    prob = 1.0 - 1.0 / (1.0 + np.exp(log))
    img = np.zeros_like(prob, dtype=np.uint8)
    img[:] = 127
    img[log <= -0.7] = 255
    img[log >= 0.7] = 0

    x_r, y_r, yaw = get_pose()
    ix = int((x_r - (grid.map_origin_xy[0] - grid.size_cells * grid.map_res / 2)) / grid.map_res)
    iy = int((y_r - (grid.map_origin_xy[1] - grid.size_cells * grid.map_res / 2)) / grid.map_res)
    if 0 <= ix < grid.size_cells and 0 <= iy < grid.size_cells:
        cv2.circle(img, (ix, iy), 2, (64,), -1)
        ex = int(ix + 8 * math.sin(yaw))
        ey = int(iy + 8 * math.cos(yaw))
        cv2.line(img, (ix, iy), (ex, ey), (64,), 1)

    vis = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    vis = cv2.flip(vis, 0)  # 上=北

    if show_window:
        cv2.imshow("Occupancy Grid (25m x 25m, 0.1m/px)", vis)
        cv2.waitKey(1)