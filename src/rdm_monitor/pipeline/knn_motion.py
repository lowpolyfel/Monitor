import cv2
import numpy as np

class KNNMotion:
    """
    Sustracción de fondo KNN:
      - Robusto a ruido y variaciones suaves
      - Similar a MOG2 pero con otro modelo de fondo
    """
    def __init__(self, history=600, dist2Threshold=400.0, detectShadows=True, learningRate=-1.0,
                 min_blob_area_px=80, morph_kernel=3):
        self.bg = cv2.createBackgroundSubtractorKNN(history=history,
                                                    dist2Threshold=dist2Threshold,
                                                    detectShadows=detectShadows)
        self.lr = learningRate
        k = morph_kernel if morph_kernel in (3,5) else 3
        self.kernel = np.ones((k,k), np.uint8)
        self.min_blob_area_px = max(0, int(min_blob_area_px))

    def initialize(self, frame_bgr):
        for _ in range(5):
            self.bg.apply(frame_bgr, learningRate=0.05)

    def _post(self, fg):
        # remover sombras (KNN suele usar 127)
        _, mask = cv2.threshold(fg, 128, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        if self.min_blob_area_px > 0:
            num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if num > 1:
                keep = np.zeros_like(mask)
                for i in range(1, num):
                    if stats[i, cv2.CC_STAT_AREA] >= self.min_blob_area_px:
                        keep[labels == i] = 255
                mask = keep
        return mask

    def step(self, frame_bgr, min_area_pct=0.001):
        fg = self.bg.apply(frame_bgr, learningRate=self.lr)
        mask = self._post(fg)
        active = float(np.count_nonzero(mask))
        total  = float(mask.size)
        frac   = active / max(1.0, total)
        score  = max(frac, min_area_pct if frac >= min_area_pct else 0.0)
        return float(score), mask
