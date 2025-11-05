import cv2
import numpy as np

class AvgMotion:
    """
    Fondo por promedio exponencial (accumulateWeighted):
      - Modelo simple y ligero
      - Reacciona a cambios de luz más lento que diff
    """
    def __init__(self, alpha=0.02, thr=18, min_blob_area_px=80, morph_kernel=3):
        self.alpha = float(alpha)
        self.thr = int(thr)
        self.bg = None
        k = morph_kernel if morph_kernel in (3,5) else 3
        self.kernel = np.ones((k,k), np.uint8)
        self.min_blob_area_px = max(0, int(min_blob_area_px))

    def initialize(self, frame_bgr):
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        self.bg = g.astype(np.float32)

    def step(self, frame_bgr, min_area_pct=0.001):
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        cv2.accumulateWeighted(g, self.bg, self.alpha)
        bg8 = cv2.convertScaleAbs(self.bg)
        diff = cv2.absdiff(g, bg8)
        _, mask = cv2.threshold(diff, self.thr, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        if self.min_blob_area_px > 0:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num > 1:
                keep = np.zeros_like(mask)
                for i in range(1, num):
                    if stats[i, cv2.CC_STAT_AREA] >= self.min_blob_area_px:
                        keep[labels == i] = 255
                mask = keep
        active = float(np.count_nonzero(mask))
        total  = float(mask.size)
        frac   = active / max(1.0, total)
        score  = max(frac, min_area_pct if frac >= min_area_pct else 0.0)
        return float(score), mask
