from typing import Tuple


def world_to_map(x: float, y: float, origin_xy: Tuple[float, float], res: float, size_cells: int):
    """世界坐标 (x,y) -> 栅格索引 (ix,iy)。地图以 origin 为地图中心。"""
    ox, oy = origin_xy
    ix = int((x - (ox - size_cells * res / 2)) / res)
    iy = int((y - (oy - size_cells * res / 2)) / res)
    return ix, iy

def bresenham(x0, y0, x1, y1):
    """整数格子上的 Bresenham 直线（包含起终点）。"""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        points.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy
    return points