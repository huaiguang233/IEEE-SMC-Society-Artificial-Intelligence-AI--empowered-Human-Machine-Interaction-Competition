import math
import numpy as np
from config import (
    R_CR_FIXED, CAM_T_R,          # 相机->机器人 外参（旋转、平移）
    DEPTH_SAMPLE_STRIP,           # 取 bbox 底部一条带状区域采样深度的比例
    DEPTH_MIN_VALID, DEPTH_MAX_VALID
)

# ------------------------------
# 读取 Webots 深度相机的一帧
# ------------------------------
def get_depth_from_cam(cam_depth):
    if cam_depth is None:
        return None, 0, 0
    buf = cam_depth.getRangeImage()
    if buf is None:
        return None, 0, 0
    w = cam_depth.getWidth()
    h = cam_depth.getHeight()
    depth = np.array(buf, dtype=np.float32).reshape((h, w))
    return depth, w, h

# --------------------------------------------
# 由水平 FOV 反推 pinhole 内参 (fx, fy, cx, cy)
# Webots Camera/RangeFinder 的 getFov() 返回水平 FOV (rad)
# --------------------------------------------
def camera_intrinsics_from_fov(width, height, fov_h):
    fx = width / (2.0 * math.tan(fov_h / 2.0))
    fov_v = 2.0 * math.atan((height / width) * math.tan(fov_h / 2.0))
    fy = height / (2.0 * math.tan(fov_v / 2.0))
    cx = width * 0.5
    cy = height * 0.5
    return fx, fy, cx, cy

# ------------------------------------------------
# 像素 + 深度 → 相机坐标系 3D 点 (单位: 米)
# 相机原生轴约定（Webots）：
#   Xc 向右、Yc 向下、Zc 向前
# ------------------------------------------------
def pixel_to_cam3d(u, v, depth_img, fov_h):
    H, W = depth_img.shape[:2]
    fx, fy, cx, cy = camera_intrinsics_from_fov(W, H, fov_h)

    # 3x3 小窗口中值滤波以提升鲁棒性
    u = int(np.clip(u, 1, W - 2))
    v = int(np.clip(v, 1, H - 2))
    win = depth_img[v-1:v+2, u-1:u+2].astype(np.float32)
    finite = np.isfinite(win)
    d = float(np.median(win[finite])) if finite.any() else float('nan')
    if (not np.isfinite(d)) or (d < DEPTH_MIN_VALID) or (d > DEPTH_MAX_VALID):
        return None

    # 反投影到相机系（保持原生轴：X右、Y下、Z前）
    x = (u - cx) / fx * d
    y = (v - cy) / fy * d
    z = d
    return np.array([x, y, z], dtype=np.float32)

# ------------------------------------------------
# 航向角（与建图一致）：
#   - compass.getValues() = [nx, ny, nz] 为“北方向在机器人系”的分量
#   - 机器人系：x 前、y 左、z 上
#   - yaw = atan2(ny, nx)，使得：yaw=0 指向北(+Y)，逆时针为正
# ------------------------------------------------
def robot_yaw_from_compass(values_xyz):
    nx, ny, nz = values_xyz
    return math.atan2(ny, nx)

# ------------------------------------------------
# 机器人 -> 世界 旋转矩阵（与建图一致）
# 基向量（列向量）在世界系的表达：
#   e_fx(前) = [ sin(yaw),  cos(yaw), 0]
#   e_ly(左) = [-cos(yaw),  sin(yaw), 0]
#   e_uz(上) = [        0,         0, 1]
# ------------------------------------------------
def Rwr_from_yaw_north(yaw):
    s, c = math.sin(yaw), math.cos(yaw)
    return np.array([
        [ s, -c, 0],
        [ c,  s, 0],
        [ 0,  0, 1]
    ], dtype=np.float32)

# ------------------------------------------------
# 核心：由检测框 + 深度图 + 外参 + 位姿，恢复世界坐标
# 输入/输出：
#   bbox: (x, y, w, h) 像素框（左上角 + 宽高）
#   depth_img: HxW float32 (米)
#   fov_h: 水平 FOV (弧度)
#   gps_xyz: 机器人世界位置 (rx, ry, rz)
#   compass_values: [nx, ny, nz]
#   R_cr: 相机->机器人 旋转（默认来自 config.R_CR_FIXED）
#   T_cr: 相机->机器人 平移（默认来自 config.CAM_T_R）
# 返回：
#   world(np.ndarray[3]) | None, status(str)
# ------------------------------------------------
def world_from_detection(bbox, depth_img, fov_h, gps_xyz, compass_values,
                         R_cr=R_CR_FIXED, T_cr=CAM_T_R):
    if depth_img is None or depth_img.size == 0:
        return None, "no depth"

    x, y, bw, bh = bbox
    H, W = depth_img.shape[:2]

    # 在 bbox 底部取一条带状区域做多点深度采样（对高柱体更稳）
    strip_h = max(3, int(bh * DEPTH_SAMPLE_STRIP))
    y0 = int(np.clip(y + bh - strip_h, 0, H - 1))
    y1 = int(np.clip(y + bh,           0, H - 1))

    xs = np.linspace(x + bw * 0.2, x + bw * 0.8, num=9)
    ys = np.linspace(y0, y1, num=5)

    pts_cam = []
    for yy in ys:
        for xx in xs:
            p = pixel_to_cam3d(int(xx), int(yy), depth_img, fov_h)
            if p is not None:
                pts_cam.append(p)

    if len(pts_cam) == 0:
        return None, "invalid depth samples"

    # 相机系中值点 → 机器人系 → 世界系
    Pc = np.median(np.stack(pts_cam, axis=0), axis=0)   # camera frame
    Pr = (R_cr @ Pc) + T_cr                              # robot frame
    yaw = robot_yaw_from_compass(compass_values)
    R_wr = Rwr_from_yaw_north(yaw)                       # world <- robot
    Pw = R_wr @ Pr                                       # world (相对机器人原点)

    rx, ry, rz = gps_xyz                                 # 机器人世界位置
    world = np.array([rx + Pw[0], ry + Pw[1], rz + Pw[2]], dtype=np.float32)
    return world, "ok"

# ------------------------------------------------
# 便捷包装：直接从 Webots 设备取数并计算
# ------------------------------------------------
def world_from_devices(bbox, cam_depth, gps, compass,
                       R_cr=R_CR_FIXED, T_cr=CAM_T_R):
    depth_img, dw, dh = get_depth_from_cam(cam_depth)
    if depth_img is None:
        return None, "missing depth"
    fov_h = cam_depth.getFov()
    gps_xyz = gps.getValues()
    compass_values = compass.getValues()
    return world_from_detection(bbox, depth_img, fov_h, gps_xyz, compass_values, R_cr, T_cr)
