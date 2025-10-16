# navigation.py
import math, heapq
from typing import List, Tuple, Optional
from config import FWD_SPEED, TURN_SPEED
from utils import world_to_map

def plan_path_to_world(grid, get_pose, target_xy, goal_tol_m=0.10, occ_th=0.7, inflate_m=0.20):
    x_r, y_r, _ = get_pose()
    sx, sy = world_to_map(x_r, y_r, grid.map_origin_xy, grid.map_res, grid.size_cells)
    gx, gy = world_to_map(target_xy[0], target_xy[1], grid.map_origin_xy, grid.map_res, grid.size_cells)

    if not (0 <= sx < grid.size_cells and 0 <= sy < grid.size_cells):
        return None, "start out of map"
    if not (0 <= gx < grid.size_cells and 0 <= gy < grid.size_cells):
        return None, "goal out of map"

    tol_cells = max(0, int(math.floor(goal_tol_m / grid.map_res)))
    goal_cells = []
    for dx in range(-tol_cells, tol_cells + 1):
        for dy in range(-tol_cells, tol_cells + 1):
            ix, iy = gx + dx, gy + dy
            if 0 <= ix < grid.size_cells and 0 <= iy < grid.size_cells:
                if not grid.inflated_is_blocked(ix, iy, occ_th, inflate_m):
                    goal_cells.append((ix, iy))
    if not goal_cells:
        return None, "no free cell within tolerance"
    goal_set = set(goal_cells)

    def h(ix, iy):
        return min(math.hypot(ix - gx2, iy - gy2) for (gx2, gy2) in goal_cells)

    openpq = []
    gscore = {(sx, sy): 0.0}
    came   = {}
    heapq.heappush(openpq, (h(sx, sy), 0.0, (sx, sy)))
    closed = set()
    found_goal = None

    nbrs = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
            (-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),(1,-1,math.sqrt(2)),(1,1,math.sqrt(2))]

    while openpq:
        f, g, cur = heapq.heappop(openpq)
        if cur in closed:
            continue
        closed.add(cur)
        if cur in goal_set:
            found_goal = cur
            break
        cx, cy = cur
        for dx, dy, step in nbrs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < grid.size_cells and 0 <= ny < grid.size_cells):
                continue
            if grid.inflated_is_blocked(nx, ny, occ_th, inflate_m):
                continue
            step_cost = step * grid.grid_cost(nx, ny, occ_th)
            ng = g + step_cost
            if ng < gscore.get((nx, ny), float('inf')):
                gscore[(nx, ny)] = ng
                came[(nx, ny)] = (cx, cy)
                nf = ng + h(nx, ny)
                heapq.heappush(openpq, (nf, ng, (nx, ny)))

    if found_goal is None:
        return None, "no path found"

    # 回溯 + 稀疏化
    path_cells = []
    node = found_goal
    while node != (sx, sy):
        path_cells.append(node)
        node = came[node]
    path_cells.append((sx, sy))
    path_cells.reverse()

    simplified = []
    prev = path_cells[0]
    simplified.append(prev)
    for p in path_cells[1:]:
        if not grid.has_line_of_sight(simplified[-1], p, occ_th, inflate_m):
            if prev != simplified[-1]:
                simplified.append(prev)
        prev = p
    if simplified[-1] != path_cells[-1]:
        simplified.append(path_cells[-1])

    path_world = [grid.map_to_world(ix, iy) for (ix, iy) in simplified]
    print(path_world)
    return path_world, f"path found: {len(path_world)} waypoints"

def follow_path_step(drive, get_pose, nav_path, nav_i, wp_tol, nav_goal_tol):
    """
    极简 Pure Pursuit（不避障）。内置方向补偿开关，快速把“前/后、左/右”调正。

    需要: drive.set_speed(left, right), get_pose()->(x,y,yaw)
    依赖常量: FWD_SPEED, TURN_SPEED
    返回: (still_navigating: bool, nav_i: int)
    """
    if not nav_path or nav_i >= len(nav_path):
        return False, nav_i

    # === 方向补偿开关（如方向不对，改这三个量就能对齐） ===
    FWD_SIGN  = +1   # 前进为正；若正速度在你机器上是后退，改成 -1
    TURN_SIGN = +1   # 正转向=左转(CCW)；若显得反了，改成 -1
    SWAP_LR   = False  # 若 set_speed 的参数顺序其实是 (right,left)，改 True

    def send(l, r):
        # 统一做补偿
        l *= FWD_SIGN
        r *= FWD_SIGN
        # 轮序交换
        if SWAP_LR:
            l, r = r, l
        drive.set_speed(l, r)

    # === 位姿与到达判定 ===
    x, y, yaw = get_pose()
    gx, gy = nav_path[-1]
    if (gx - x)**2 + (gy - y)**2 < (nav_goal_tol ** 2):
        drive.stop()
        return False, len(nav_path)

    tx, ty = nav_path[nav_i]
    if (tx - x)**2 + (ty - y)**2 < (wp_tol ** 2):
        nav_i += 1
        if nav_i >= len(nav_path):
            drive.stop()
            return False, nav_i
        tx, ty = nav_path[nav_i]

    # === 选前视点（Ld 简单自适应） ===
    Ld = max(0.20, min(1.00, 2.0 * wp_tol))
    look_x, look_y = nav_path[-1]
    for i in range(nav_i, len(nav_path)):
        px, py = nav_path[i]
        if (px - x)**2 + (py - y)**2 >= (Ld ** 2):
            look_x, look_y = px, py
            break

    # === 世界->机体系（前正、左正），全程不用角度差，避免 ±π 抖动 ===
    dx, dy = look_x - x, look_y - y
    left_body  = -math.sin(yaw) * dx + math.cos(yaw) * dy  # 左(+)/右(-)
    fwd_body   =  math.cos(yaw) * dx + math.sin(yaw) * dy  # 前(+)/后(-)
    Ld_eff = max(0.15, (left_body * left_body + fwd_body * fwd_body) ** 0.5)

    # === Pure Pursuit 曲率与差速 ===
    kappa = 2.0 * left_body / (Ld_eff * Ld_eff)     # 左正右负
    v_scale = max(0.25, min(1.0, (abs(gx - x) + abs(gy - y)) / (3.0 * nav_goal_tol)))
    v = v_scale * FWD_SPEED
    turn = kappa * v                                 # 角速度 ~ v*kappa
    # 把“正转向=左转”映射到轮差；若反了，用 TURN_SIGN 翻转
    turn *= TURN_SIGN
    # 裁剪，避免过猛
    if turn >  TURN_SPEED: turn =  TURN_SPEED
    if turn < -TURN_SPEED: turn = -TURN_SPEED

    left  = v - turn
    right = v + turn
    send(left, right)
    return True, nav_i


