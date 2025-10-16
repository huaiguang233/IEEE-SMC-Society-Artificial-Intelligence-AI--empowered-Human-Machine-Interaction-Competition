# -*- coding: utf-8 -*-
import numpy as np



# -*- coding: utf-8 -*-
import numpy as np

# ---------- 颜色/连通域参数 ----------
LOWER_RED_1 = np.array([0,   150, 80], dtype=np.uint8)
UPPER_RED_1 = np.array([3,   255, 255], dtype=np.uint8)
LOWER_RED_2 = np.array([177, 150, 80], dtype=np.uint8)
UPPER_RED_2 = np.array([179, 255, 255], dtype=np.uint8)

MIN_AREA_PX        = 300
ASPECT_MIN         = 3
HEIGHT_FRAC_MIN    = 0.25
SOLIDITY_MIN       = 0.9
PURITY_MIN         = 0.8

CONF_EMA_ALPHA     = 0.55
W_ASPECT, W_AREA, W_PURITY = 0.5, 0.2, 0.3

# ---------- 相机外参（相机相对机器人坐标系） ----------
# 机器人系：Xr 前、Yr 左、Zr 上；相机系：Xc 右、Yc 下、Zc 前
R_CR_FIXED = np.array([[ 0,  0,  1],
                       [-1,  0,  0],
                       [ 0, -1,  0]], dtype=np.float32)
CAM_T_R = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # m

# ---------- 深度采样 ----------
DEPTH_SAMPLE_STRIP = 0.25
DEPTH_MIN_VALID    = 0.05
DEPTH_MAX_VALID    = 10.0

SHOW_WINDOW = True
SAVE_EVERY_S = 2.0

# ===================== 配置 =====================
MAP_SIZE_M   = 25.0       # 地图边长（米）
MAP_RES_M    = 0.10       # 地图分辨率（米/格）
LOG_FREE     = -0.35      # 自由格对数几率增量
LOG_OCC      = +0.10      # 占据格对数几率增量
LOG_MIN      = -4.0       # 对数几率下限（更自由）
LOG_MAX      = +4.0       # 对数几率上限（更占据）
HIT_EPS      = 0.12       # 命中延伸补偿（米），避免缺口

# 运动参数
MAX_SPEED  = 25.0     # 电机目标速度上限（差速两侧都会被 clamp）
FWD_SPEED  = 10.0     # 前进速度
TURN_SPEED = 4.0     # 原地转向速度

# 探索与避障
SAFE_RANGE_M        = 0.35  # 正前方安全距离
GAP_MIN_WIDTH_RAD   = 0.35  # 认为“可通过缝隙”的最小角宽（未直接用，可扩展）
FRONT_ARC_RAD       = 0.70  # 前向避障检测角（±）
SCAN_SUBSAMPLE_EVERY = 2    # 激光线束下采样（每 N 束取 1 束，加速）

# 受害者扫描调度（基于 Webots 仿真时间）
SPIN_PERIOD_S   = 20.0  # 每隔 20 秒触发一次扫描
SPIN_DURATION_S = 3.0   # 扫描持续 3 秒（原地旋转）