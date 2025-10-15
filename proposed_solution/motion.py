# motion.py
from config import MAX_SPEED, FWD_SPEED, TURN_SPEED

class DifferentialDrive:
    def __init__(self, mot_fl, mot_fr, mot_rl, mot_rr):
        self.mot_fl, self.mot_fr = mot_fl, mot_fr
        self.mot_rl, self.mot_rr = mot_rl, mot_rr
        for m in [mot_fl, mot_fr, mot_rl, mot_rr]:
            m.setPosition(float('inf'))
            m.setVelocity(0.0)

    def set_speed(self, l, r):
        l = max(-MAX_SPEED, min(MAX_SPEED, l))
        r = max(-MAX_SPEED, min(MAX_SPEED, r))
        self.mot_fl.setVelocity(l); self.mot_rl.setVelocity(l)
        self.mot_fr.setVelocity(r); self.mot_rr.setVelocity(r)

    def forward(self, v=FWD_SPEED): self.set_speed(v, v)
    def turn_left(self, v=TURN_SPEED): self.set_speed(-v, v)
    def turn_right(self, v=TURN_SPEED): self.set_speed(v, -v)
    def stop(self): self.set_speed(0.0, 0.0)
