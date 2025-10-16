# spin.py
from config import SPIN_DURATION_S, SPIN_PERIOD_S, TURN_SPEED

class SpinScheduler:
    def __init__(self, now: float):
        self.spin_end_time  = now + SPIN_DURATION_S
        self.next_spin_time = now + SPIN_PERIOD_S
        self.spin_turn_dir  = 1  # +1 左转，-1 右转

    def spinning_now(self, t: float) -> bool:
        if not (self.spin_end_time is not None and t < self.spin_end_time):
            if t >= self.next_spin_time:
                self.spin_end_time = t + SPIN_DURATION_S
                self.next_spin_time = t + SPIN_PERIOD_S
                self.spin_turn_dir *= -1
        return (self.spin_end_time is not None) and (t < self.spin_end_time)
