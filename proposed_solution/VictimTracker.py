import math
from typing import List, Tuple, Optional

class VictimTracker:
    """
    在线聚合重复检测的受害者位置（仅 x,y）。
    - 累计命中 >= min_hits_confirm 时“冻结”进入 confirmed 列表
    - 冻结后位置不再变化，后续相似观测被忽略
    """
    def __init__(self,
                 gate_m: float = 1.0,          # 关联门限（米）：距离 <= gate_m 视为同一目标
                 ema_alpha: float = 0.3,       # EMA 系数：平滑更新
                 merge_dist_m: float = 0.6,    # 合并阈值：彼此距离 < merge_dist_m 则合并
                 min_hits_confirm: int = 10    # 命中次数阈值：≥该值后冻结为确认目标
                 ):
        self.gate_m = gate_m
        self.ema_alpha = ema_alpha
        self.merge_dist_m = merge_dist_m
        self.min_hits_confirm = min_hits_confirm

        # 未确认的临时跟踪目标
        self._tracks: List[dict] = []
        # ✅ 已确认（冻结）的最终受害者位置
        self._confirmed: List[Tuple[float, float]] = []

    # ------------------ 内部辅助函数 ------------------
    @staticmethod
    def _dist2(x1, y1, x2, y2) -> float:
        dx, dy = x1 - x2, y1 - y2
        return dx*dx + dy*dy

    def _find_nearest_idx(self, x: float, y: float) -> Optional[int]:
        """返回最近未确认条目的索引（不做门限判断）。"""
        if not self._tracks:
            return None
        best_i, best_d2 = None, float("inf")
        for i, tr in enumerate(self._tracks):
            d2 = self._dist2(x, y, tr["x"], tr["y"])
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return best_i

    def _maybe_merge_close(self):
        """合并过近的临时目标。"""
        if len(self._tracks) < 2:
            return
        merged = [False] * len(self._tracks)
        new_tracks = []
        for i in range(len(self._tracks)):
            if merged[i]:
                continue
            base = self._tracks[i]
            bx, by = base["x"], base["y"]
            b_hits = base["hits"]
            for j in range(i + 1, len(self._tracks)):
                if merged[j]:
                    continue
                other = self._tracks[j]
                if self._dist2(bx, by, other["x"], other["y"]) <= self.merge_dist_m**2:
                    # 合并（命中次数作权重平均）
                    tot_hits = b_hits + other["hits"]
                    if tot_hits > 0:
                        bx = (bx*b_hits + other["x"]*other["hits"]) / tot_hits
                        by = (by*b_hits + other["y"]*other["hits"]) / tot_hits
                    b_hits = tot_hits
                    merged[j] = True
            new_tracks.append({"x": bx, "y": by, "hits": b_hits})
            merged[i] = True
        self._tracks = new_tracks

    # ------------------ 更新逻辑 ------------------
    def update_with_observation(self, x: float, y: float):
        """
        加入一次观测：
        - 若与已确认目标接近 → 忽略
        - 若与临时 track 匹配 → EMA 更新 + hits++
        - 若 hits >= min_hits_confirm → 冻结加入 confirmed
        - 否则创建新 track
        """
        # 1️⃣ 如果与 confirmed 目标太近 → 忽略
        for (cx, cy) in self._confirmed:
            if math.sqrt(self._dist2(x, y, cx, cy)) <= self.gate_m:
                return  # 已确认目标，忽略新观测

        # 2️⃣ 更新未确认 track
        if not self._tracks:
            self._tracks.append({"x": x, "y": y, "hits": 1})
            return

        idx = self._find_nearest_idx(x, y)
        tr = self._tracks[idx]
        if math.sqrt(self._dist2(x, y, tr["x"], tr["y"])) <= self.gate_m:
            # 同一目标 → EMA 平滑更新
            a = self.ema_alpha
            tr["x"] = a * x + (1.0 - a) * tr["x"]
            tr["y"] = a * y + (1.0 - a) * tr["y"]
            tr["hits"] += 1

            # 若累计命中达到阈值 → 冻结
            if tr["hits"] >= self.min_hits_confirm:
                self._confirmed.append((tr["x"], tr["y"]))
                self._tracks.pop(idx)  # 从临时列表移除
        else:
            # 新目标
            self._tracks.append({"x": x, "y": y, "hits": 1})

        # 合并过近的临时条目
        self._maybe_merge_close()

    # ------------------ 查询接口 ------------------
    def get_all_candidates(self) -> List[Tuple[float, float]]:
        """返回所有未确认的临时候选点（调试用）。"""
        return [(tr["x"], tr["y"]) for tr in self._tracks]

    def get_confirmed(self) -> List[Tuple[float, float]]:
        """返回已确认（冻结）的受害者位置列表。"""
        return list(self._confirmed)
