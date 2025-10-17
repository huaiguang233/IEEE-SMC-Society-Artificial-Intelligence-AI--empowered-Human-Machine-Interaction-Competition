"""
Co-Op Lidar Mapping + Frontier Sharing (统一原点 / 白黑共享 / 去重探索)
- 统一全局地图原点，世界坐标直接落格融合
- 广播本帧 free(白)/occ(黑) 点 + frontier 目标 claim，减少重复探索
- 简单导航：有目标时朝目标飞行 + 雷达避障；无目标时缝隙跟随
"""

import math
import json
from collections import deque
from typing import Tuple, List, Dict, Optional

import numpy as np
import cv2
from controller import Robot

# ===================== 地图 / 通信基线 =====================
MAP_SIZE_M   = 25.0
MAP_RES_M    = 0.10
LOG_FREE     = -0.35
LOG_OCC      = +0.10
LOG_MIN      = -4.0
LOG_MAX      = +4.0
HIT_EPS      = 0.12

# 👉 全队统一的全局地图原点（务必所有机器人一致）
GLOBAL_MAP_ORIGIN_XY = (0.0, 0.0)

SHOW_WINDOW  = True
SAVE_EVERY_S = 2.0

# ===================== 运动 / 探索 =====================
MAX_SPEED  = 30.0
FWD_SPEED  = 15.0
TURN_SPEED = 10.0

SAFE_RANGE_M         = 0.50
FRONT_ARC_RAD        = 0.70
SCAN_SUBSAMPLE_EVERY = 2
FREE_THIN_STRIDE     = 3

MAX_POINTS_PER_TX     = 2000     # 地图更新包点数上限
BROADCAST_EVERY_STEPS = 1
CLAIM_TTL_S           = 3.0      # 目标占用（秒）
CLAIM_DISTANCE_M      = 0.8      # 认为“同一目标”的距离阈值

# 通信设备名 + 频道（推荐固定频道避免串台）
EMITTER_NAME  = "robot to robot emitter"
RECEIVER_NAME = "robot to robot receiver"
CHANNEL_ID    = 42

# ===================== 工具函数 =====================
def world_to_map(x: float, y: float, origin_xy: Tuple[float, float], res: float, size_cells: int):
    ox, oy = origin_xy
    ix = int((x - (ox - size_cells * res / 2)) / res)
    iy = int((y - (oy - size_cells * res / 2)) / res)
    return ix, iy

def map_to_world(ix: int, iy: int, origin_xy: Tuple[float, float], res: float, size_cells: int):
    ox, oy = origin_xy
    x = (ox - size_cells * res / 2) + (ix + 0.5) * res
    y = (oy - size_cells * res / 2) + (iy + 0.5) * res
    return x, y

def bresenham(x0, y0, x1, y1):
    pts = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        pts.append((x, y))
        if x == x1 and y == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x += sx
        if e2 <= dx: err += dx; y += sy
    return pts

def l2(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ===================== 控制器 =====================
class CoopFrontierMapper:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.name = self.robot.getName()

        self._init_devices()

        # 地图（统一原点）
        self.map_res   = MAP_RES_M
        self.map_size  = MAP_SIZE_M
        self.size_cells = int(self.map_size / self.map_res)
        self.map_log = np.zeros((self.size_cells, self.size_cells), dtype=np.float32)
        self.map_origin_xy = GLOBAL_MAP_ORIGIN_XY  # ✅ 全队一致
        self.last_save_t = 0.0

        # 通信 / 可视化
        self.step_counter = 0
        self.teammates: Dict[str, dict] = {}
        self.trail_len = 2000
        self.claims: Dict[str, dict] = {}  # rid -> {"xy":(x,y), "cost":float, "ts":float}

        # 探索目标
        self.goal_xy: Optional[Tuple[float,float]] = None

        print(f"[{self.name}] CoopFrontierMapper initialized (global origin={self.map_origin_xy}).")

    # ---------- 设备 ----------
    def _init_devices(self):
        # 电机
        self.mot_fl = self.robot.getDevice("fl_wheel_joint")
        self.mot_fr = self.robot.getDevice("fr_wheel_joint")
        self.mot_rl = self.robot.getDevice("rl_wheel_joint")
        self.mot_rr = self.robot.getDevice("rr_wheel_joint")
        for m in [self.mot_fl, self.mot_fr, self.mot_rl, self.mot_rr]:
            m.setPosition(float('inf')); m.setVelocity(0.0)

        # Lidar
        self.lidar = self.robot.getDevice("laser")
        self.lidar.enable(self.timestep)
        self.lidar_res   = self.lidar.getHorizontalResolution()
        self.lidar_layers= self.lidar.getNumberOfLayers()
        self.lidar_fov   = self.lidar.getFov()
        self.lidar_min   = self.lidar.getMinRange()
        self.lidar_max   = self.lidar.getMaxRange()

        # GPS & Compass
        self.gps = self.robot.getDevice("gps"); self.gps.enable(self.timestep)
        self.compass = self.robot.getDevice("imu compass"); self.compass.enable(self.timestep)

        # 通信
        try:
            self.rx = self.robot.getDevice(RECEIVER_NAME)
            self.rx.enable(self.timestep)
            self.rx.setChannel(CHANNEL_ID)
        except:
            self.rx = None
            print(f"[{self.name}] Warning: no receiver '{RECEIVER_NAME}'")
        try:
            self.tx = self.robot.getDevice(EMITTER_NAME)
            self.tx.setChannel(CHANNEL_ID)
            try:
                self.tx.setRange(float('inf'))
            except: pass
        except:
            self.tx = None
            print(f"[{self.name}] Warning: no emitter '{EMITTER_NAME}'")

    # ---------- 位姿 / 运动 ----------
    def get_pose(self):
        x, y, z = self.gps.getValues()
        north = self.compass.getValues()  # robot frame (x fwd, y left, z up)
        yaw = math.atan2(north[1], north[0])  # 北=0，逆时针+
        return x, y, yaw

    def set_speed(self, l, r):
        l = max(-MAX_SPEED, min(MAX_SPEED, l))
        r = max(-MAX_SPEED, min(MAX_SPEED, r))
        self.mot_fl.setVelocity(l); self.mot_rl.setVelocity(l)
        self.mot_fr.setVelocity(r); self.mot_rr.setVelocity(r)

    def forward(self, v=FWD_SPEED):    self.set_speed(v, v)
    def turn_left(self, v=TURN_SPEED): self.set_speed(-v, v)
    def turn_right(self, v=TURN_SPEED):self.set_speed(v, -v)
    def stop(self):                    self.set_speed(0.0, 0.0)

    # ---------- 地图融合 ----------
    def _apply_delta_world(self, xw: float, yw: float, delta: float):
        ix, iy = world_to_map(xw, yw, self.map_origin_xy, self.map_res, self.size_cells)
        if 0 <= ix < self.size_cells and 0 <= iy < self.size_cells:
            self.map_log[iy, ix] = np.clip(self.map_log[iy, ix] + delta, LOG_MIN, LOG_MAX)

    # ---------- 通信：地图更新 + 目标 claim ----------
    def _broadcast_map_update(self, free_pts: List[Tuple[float,float]], occ_pts: List[Tuple[float,float]]):
        if not self.tx: return
        total = len(free_pts) + len(occ_pts)
        if total > MAX_POINTS_PER_TX and total > 0:
            stride = max(1, total // MAX_POINTS_PER_TX)
            free_pts = free_pts[::stride]; occ_pts = occ_pts[::stride]
        x, y, yaw = self.get_pose()
        msg = {
            "type": "map_update",
            "robot_id": self.name,
            "timestamp": self.robot.getTime(),
            "pose": [x, y, yaw],
            "free": [[round(px, 2), round(py, 2)] for (px, py) in free_pts],
            "occ":  [[round(px, 2), round(py, 2)] for (px, py) in occ_pts],
        }
        self.tx.send(json.dumps(msg))

    def _broadcast_claim(self, goal_xy: Tuple[float,float], cost: float):
        if not self.tx: return
        msg = {
            "type": "claim",
            "robot_id": self.name,
            "timestamp": self.robot.getTime(),
            "goal": [round(goal_xy[0], 2), round(goal_xy[1], 2)],
            "cost": round(cost, 3),
        }
        self.tx.send(json.dumps(msg))

    def _receive_all(self):
        if not self.rx: return
        while self.rx.getQueueLength() > 0:
            text = self.rx.getString()
            self.rx.nextPacket()
            try:
                msg = json.loads(text)
            except Exception:
                continue

            mtype = msg.get("type", "")
            if mtype == "map_update":
                rid = msg.get("robot_id", "robot")
                pose = msg.get("pose", None)
                free = msg.get("free", [])
                occ  = msg.get("occ",  [])
                ts   = float(msg.get("timestamp", 0.0))

                # 融合
                for xw, yw in free: self._apply_delta_world(float(xw), float(yw), LOG_FREE)
                for xw, yw in occ:  self._apply_delta_world(float(xw), float(yw), LOG_OCC)

                # 队友状态
                if rid not in self.teammates:
                    self.teammates[rid] = {"pose": (0,0,0,0), "trail": deque(maxlen=self.trail_len)}
                if pose and len(pose)==3:
                    self.teammates[rid]["pose"] = (float(pose[0]), float(pose[1]), float(pose[2]), ts)

                # 痕迹（稀疏）
                sample = []
                if len(free)>0: sample += free[::max(1, len(free)//120)]
                if len(occ)>0:  sample += occ[::max(1, len(occ)//120)]
                for px, py in sample:
                    self.teammates[rid]["trail"].append((float(px), float(py)))

            elif mtype == "claim":
                rid = msg.get("robot_id", "robot")
                if rid == self.name: continue
                goal = msg.get("goal", None)
                cost = float(msg.get("cost", 1e9))
                ts   = float(msg.get("timestamp", 0.0))
                if goal and len(goal)==2:
                    self.claims[rid] = {"xy": (float(goal[0]), float(goal[1])), "cost": cost, "ts": ts}

    # ---------- 扫描 / 更新 / 广播 ----------
    def update_map_with_scan_and_broadcast(self):
        x_r, y_r, yaw = self.get_pose()
        ranges = self.lidar.getRangeImage()
        if self.lidar_layers > 1:
            ranges = ranges[0:self.lidar_res]
        dtheta = self.lidar_fov / max(1, self.lidar_res - 1)
        ix0, iy0 = world_to_map(x_r, y_r, self.map_origin_xy, self.map_res, self.size_cells)

        free_pts_world: List[Tuple[float,float]] = []
        occ_pts_world:  List[Tuple[float,float]] = []

        for i in range(0, self.lidar_res, SCAN_SUBSAMPLE_EVERY):
            r = ranges[i]
            if np.isnan(r) or r <= self.lidar_min or r > self.lidar_max:
                continue
            theta = -self.lidar_fov/2.0 + i * dtheta
            beam_world = yaw + theta

            hit_dist = min(r + HIT_EPS, self.lidar_max)
            x_hit = x_r + hit_dist * math.sin(beam_world)
            y_hit = y_r + hit_dist * math.cos(beam_world)

            ix1, iy1 = world_to_map(x_hit, y_hit, self.map_origin_xy, self.map_res, self.size_cells)
            line = bresenham(ix0, iy0, ix1, iy1)

            PROTECT_CELLS = 2
            protect = set(line[-(PROTECT_CELLS + 1):-1]) if len(line) > PROTECT_CELLS else set()

            # 自由
            for k, (ix, iy) in enumerate(line[:-1]):
                if (ix, iy) in protect: continue
                if 0 <= ix < self.size_cells and 0 <= iy < self.size_cells:
                    if self.map_log[iy, ix] >= 0.8: continue
                    self.map_log[iy, ix] = np.clip(self.map_log[iy, ix] + LOG_FREE, LOG_MIN, LOG_MAX)
                    if (k % FREE_THIN_STRIDE)==0:
                        xw, yw = map_to_world(ix, iy, self.map_origin_xy, self.map_res, self.size_cells)
                        free_pts_world.append((xw, yw))

            # 命中
            if 0 <= ix1 < self.size_cells and 0 <= iy1 < self.size_cells:
                self.map_log[iy1, ix1] = np.clip(self.map_log[iy1, ix1] + LOG_OCC, LOG_MIN, LOG_MAX)
                xh, yh = map_to_world(ix1, iy1, self.map_origin_xy, self.map_res, self.size_cells)
                occ_pts_world.append((xh, yh))

        if (self.step_counter % BROADCAST_EVERY_STEPS)==0:
            self._broadcast_map_update(free_pts_world, occ_pts_world)

    # ---------- 前沿提取 / 目标选择 ----------
    def _is_free(self, iy, ix) -> bool:
        return self.map_log[iy, ix] <= -0.7
    def _is_occ(self, iy, ix) -> bool:
        return self.map_log[iy, ix] >= 0.7
    def _is_unknown(self, iy, ix) -> bool:
        v = self.map_log[iy, ix]
        return (-0.7 < v < 0.7)

    def _frontiers(self, stride:int=2) -> List[Tuple[int,int]]:
        """返回若干前沿栅格索引(iy,ix)：未知，且邻接至少一个自由格"""
        H, W = self.map_log.shape
        res: List[Tuple[int,int]] = []
        for iy in range(1, H-1, stride):
            for ix in range(1, W-1, stride):
                if not self._is_unknown(iy, ix): continue
                # 4-邻域有自由即视为前沿
                if self._is_free(iy-1, ix) or self._is_free(iy+1, ix) or self._is_free(iy, ix-1) or self._is_free(iy, ix+1):
                    res.append((iy, ix))
        return res

    def _prune_expired_claims(self):
        tnow = self.robot.getTime()
        for rid in list(self.claims.keys()):
            if tnow - self.claims[rid]["ts"] > CLAIM_TTL_S:
                self.claims.pop(rid, None)

    def _is_claimed_by_other(self, xy: Tuple[float,float], my_cost: float) -> bool:
        """若其它机器人在CLAIM_TTL_S内对相近目标提出“更优或同等（但ID更小）”的claim，则返回True"""
        self._prune_expired_claims()
        for rid, data in self.claims.items():
            goal_xy = data["xy"]; cost = data["cost"]; ts = data["ts"]
            if l2(goal_xy, xy) <= CLAIM_DISTANCE_M:
                if (cost < my_cost) or (abs(cost-my_cost) < 1e-3 and rid < self.name):
                    return True
        return False

    def _choose_frontier_goal(self) -> Optional[Tuple[float,float]]:
        x,y,yaw = self.get_pose()
        frontiers = self._frontiers(stride=2)
        if not frontiers: return None

        # 选最近的未被他人“占用”的前沿
        # 代价 = 直线距离；如被占用，则略过（或随机扰动）
        cands: List[Tuple[float,Tuple[float,float]]] = []
        for (iy, ix) in frontiers:
            wx, wy = map_to_world(ix, iy, self.map_origin_xy, self.map_res, self.size_cells)
            dist = l2((x,y), (wx,wy))
            cands.append((dist, (wx,wy)))
        cands.sort(key=lambda t: t[0])

        for dist, (wx,wy) in cands[:200]:  # 限定前200个最近点检查，加速
            # 我先假定我的成本 = 直线距离
            my_cost = dist
            if not self._is_claimed_by_other((wx,wy), my_cost):
                # 广播占用，减少撞车
                self._broadcast_claim((wx,wy), my_cost)
                return (wx,wy)
        return None

    # ---------- 导航到目标 ----------
    def _drive_to_goal(self):
        if not self.goal_xy:
            self.pick_gap_and_drive()
            return
        x,y,yaw = self.get_pose()
        gx, gy = self.goal_xy
        # 若目标被他人占用且更优 -> 放弃
        if self._is_claimed_by_other(self.goal_xy, l2((x,y), self.goal_xy)):
            self.goal_xy = None
            return

        # 到达判定
        if l2((x,y), (gx,gy)) < 0.5:
            self.goal_xy = None
            return

        # 朝向控制 + 前向避障
        desired = math.atan2(gx - x, gy - y)  # 注意：世界北=Y+，这里用 (sin,cos) 交换，所以atan2(xdiff, ydiff)
        err = (desired - yaw + math.pi) % (2*math.pi) - math.pi  # [-pi,pi]
        # 简单PD（只用P）
        w = max(-TURN_SPEED, min(TURN_SPEED, err * 8.0))
        v = FWD_SPEED
        # 前向障碍保护
        ranges = self.lidar.getRangeImage()
        if self.lidar_layers > 1:
            ranges = ranges[0:self.lidar_res]
        mid = self.lidar_res // 2
        span = int((FRONT_ARC_RAD / self.lidar_fov) * self.lidar_res)
        lo = max(0, mid - span); hi = min(self.lidar_res - 1, mid + span)
        front_min = np.nanmin(ranges[lo:hi+1])
        if np.isnan(front_min): front_min = self.lidar_max
        if front_min < SAFE_RANGE_M:
            # 让位给避障策略
            self.pick_gap_and_drive()
            return
        # 差速
        self.set_speed(v - w, v + w)

    # ---------- 可视化 ----------
    def render_map(self):
        log = self.map_log
        img = np.zeros_like(log, dtype=np.uint8); img[:] = 127
        img[log <= -0.7] = 255
        img[log >= 0.7]  = 0

        # 画前沿（淡灰）
        for (iy, ix) in self._frontiers(stride=3):
            img[iy, ix] = 180 if img[iy,ix]==127 else img[iy,ix]

        # 本机
        x_r, y_r, yaw = self.get_pose()
        ix, iy = world_to_map(x_r, y_r, self.map_origin_xy, self.map_res, self.size_cells)
        if 0 <= ix < self.size_cells and 0 <= iy < self.size_cells:
            cv2.circle(img, (ix, iy), 2, (60,), -1)
            ex = int(ix + 8 * math.sin(yaw)); ey = int(iy + 8 * math.cos(yaw))
            cv2.line(img, (ix, iy), (ex, ey), (80,), 1)

        # 目标
        if self.goal_xy:
            gx, gy = self.goal_xy
            gix, giy = world_to_map(gx, gy, self.map_origin_xy, self.map_res, self.size_cells)
            if 0 <= gix < self.size_cells and 0 <= giy < self.size_cells:
                cv2.circle(img, (gix, giy), 3, (20,), 1)

        # 队友
        for rid, info in self.teammates.items():
            px, py, pyaw, _ = info["pose"]
            ixt, iyt = world_to_map(px, py, self.map_origin_xy, self.map_res, self.size_cells)
            if 0 <= ixt < self.size_cells and 0 <= iyt < self.size_cells:
                cv2.circle(img, (ixt, iyt), 2, (100,), -1)
                ex = int(ixt + 7 * math.sin(pyaw)); ey = int(iyt + 7 * math.cos(pyaw))
                cv2.line(img, (ixt, iyt), (ex, ey), (100,), 1)
            for (wx, wy) in list(info["trail"])[-400:]:
                ixp, iyp = world_to_map(wx, wy, self.map_origin_xy, self.map_res, self.size_cells)
                if 0 <= ixp < self.size_cells and 0 <= iyp < self.size_cells:
                    if img[iyp, ixp] == 127: img[iyp, ixp] = 200

        vis = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        vis = cv2.flip(vis, 0)
        if SHOW_WINDOW:
            cv2.imshow(f"Grid — {self.name}", vis); cv2.waitKey(1)
        t = self.robot.getTime()
        if t - self.last_save_t > SAVE_EVERY_S:
            cv2.imwrite(f"map_{self.name}.png", vis); self.last_save_t = t

    # ---------- 简易避障（无目标时） ----------
    def pick_gap_and_drive(self):
        ranges = self.lidar.getRangeImage()
        if self.lidar_layers > 1: ranges = ranges[0:self.lidar_res]
        mid = self.lidar_res // 2
        span = int((FRONT_ARC_RAD / self.lidar_fov) * self.lidar_res)
        lo = max(0, mid - span); hi = min(self.lidar_res - 1, mid + span)
        front_min = np.nanmin(ranges[lo:hi+1]);
        if np.isnan(front_min): front_min = self.lidar_max
        if front_min < SAFE_RANGE_M:
            thr = SAFE_RANGE_M
            good = np.array([(not np.isnan(r) and r > thr) for r in ranges])
            best_len = 0; best_start = None; cur_len = 0; cur_start = 0
            for i, ok in enumerate(good):
                if ok:
                    if cur_len == 0: cur_start = i
                    cur_len += 1
                    if cur_len > best_len:
                        best_len = cur_len; best_start = cur_start
                else: cur_len = 0
            if best_len > 0:
                center_idx = best_start + best_len // 2
                theta = -self.lidar_fov/2 + center_idx * (self.lidar_fov / max(1, self.lidar_res - 1))
                if abs(theta) < 0.05: self.forward(max(1.0, FWD_SPEED*0.6))
                elif theta > 0:       self.turn_right()
                else:                 self.turn_left()
            else:
                self.turn_left()
        else:
            self.forward(FWD_SPEED)

    # ---------- 主循环 ----------
    def run(self):
        print(f"[{self.name}] Cooperative mapping started.")
        while self.robot.step(self.timestep) != -1:
            self.step_counter += 1
            # 收消息（地图+claim）
            self._receive_all()
            # 建图并广播更新
            self.update_map_with_scan_and_broadcast()
            # 选择/维护目标（基于全队联合地图）
            if self.goal_xy is None:
                self.goal_xy = self._choose_frontier_goal()
            # 执行导航或避障
            self._drive_to_goal()
            # 可视化
            self.render_map()

def main():
    ctrl = CoopFrontierMapper()
    ctrl.run()

if __name__ == "__main__":
    main()
