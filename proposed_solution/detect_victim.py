import cv2
import numpy as np
from config import (
    LOWER_RED_1, UPPER_RED_1, LOWER_RED_2, UPPER_RED_2,
    MIN_AREA_PX, ASPECT_MIN, HEIGHT_FRAC_MIN, SOLIDITY_MIN, PURITY_MIN,
    W_ASPECT, W_AREA, W_PURITY, CONF_EMA_ALPHA,
)

def get_bgr_from_cam(cam_rgb):
    if cam_rgb is None:
        return None, 0, 0
    img = cam_rgb.getImage()
    if not img:
        return None, 0, 0
    w = cam_rgb.getWidth()
    h = cam_rgb.getHeight()
    bgra = np.frombuffer(img, dtype=np.uint8).reshape((h, w, 4))
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    return bgr, w, h


def red_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1) | \
           cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
    mask = cv2.medianBlur(mask, 5)
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k3, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5, iterations=1)
    return mask

def purity(mask, cnt):
    fill = np.zeros_like(mask)
    cv2.drawContours(fill, [cnt], -1, 255, thickness=cv2.FILLED)
    inside = (fill == 255)
    if inside.sum() == 0:
        return 0.0
    return float((mask[inside] > 0).sum()) / float(inside.sum())


def _near_left_right_edge(x, bw, w, margin_frac=0.05):

    margin = int(w * margin_frac)
    left_hits  = x <= margin
    right_hits = (x + bw) >= (w - margin)
    return left_hits or right_hits


def detect_victim_on_bgr(bgr, conf_prev=0.0):

    if bgr is None:
        return False, None, (0,0,0,0), {"conf_raw":0.0,"purity":0.0,"conf":conf_prev,"img_size":(0,0)}, conf_prev

    h, w = bgr.shape[:2]
    mask = red_mask(bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img_area = float(w * h)
    best = None
    best_score = -1.0
    best_purity = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)


        # if _near_left_right_edge(x, bw, w, margin_frac=0.05):
        #     continue

        aspect = bh / max(1, bw)
        if aspect < ASPECT_MIN:
            continue
        if bh < HEIGHT_FRAC_MIN * h:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 1e-6:
            continue
        solid = float(area) / float(hull_area)
        if solid < SOLIDITY_MIN:
            continue

        p = purity(mask, cnt)
        if p < PURITY_MIN:
            continue

        area_ratio = area / img_area
        aspect_norm = np.clip((aspect - ASPECT_MIN)/(ASPECT_MIN + 1.0), 0.0, 1.0)
        score = (W_ASPECT*aspect_norm +
                 W_AREA  *np.clip(area_ratio * 6.0, 0.0, 1.0) +
                 W_PURITY*np.clip(p, 0.0, 1.0))

        if score > best_score:
            best_score = score
            best = (x, y, bw, bh, aspect, solid, p)

    if best is None:
        conf_next = (1 - CONF_EMA_ALPHA) * conf_prev
        metrics = {"conf_raw":0.0, "purity":0.0, "conf":conf_next, "img_size":(w,h)}
        return False, None, (0,0,0,0), metrics, conf_next

    x, y, bw, bh, aspect, solid, p = best
    cx, cy = x + bw // 2, y + bh // 2
    conf_raw = float(np.clip(best_score, 0.0, 1.0))
    conf_next = CONF_EMA_ALPHA * conf_raw + (1 - CONF_EMA_ALPHA) * conf_prev
    metrics = {
        "conf_raw": conf_raw,
        "purity": p,
        "conf": conf_next,
        "img_size": (w, h),
        "aspect": aspect,
        "solidity": solid
    }
    return True, (cx, cy), (x, y, bw, bh), metrics, conf_next

def detect_victim_from_cam(cam_rgb, conf_prev=0.0):
    bgr, w, h = get_bgr_from_cam(cam_rgb)
    return detect_victim_on_bgr(bgr, conf_prev)

