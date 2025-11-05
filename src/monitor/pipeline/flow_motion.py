import cv2
import numpy as np

class FlowMotion:
    """
    Flujo óptico (Farnebäck):
      - Magnitud de movimiento por pixel
      - Más costoso en CPU pero sensible a movimientos suaves
    """
    def __init__(self, flow_thr=0.7, min_blob_area_px=80, morph_kernel=3):
        self.prev_gray = None
        self.flow_thr = float(flow_thr)
        k = morph_kernel if morph_kernel in (3,5) else 3
        self.kernel = np.ones((k,k), np.uint8)
        self.min_blob_area_px = max(0, int(min_blob_area_px))

    def initialize(self, frame_bgr):
        self.prev_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    def _mask_from_mag(self, mag):
        mask = (mag >= self.flow_thr).astype(np.uint8) * 255
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
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        mask = self._mask_from_mag(mag)
        active = float(np.count_nonzero(mask))
        total  = float(mask.size)
        frac   = active / max(1.0, total)
        score  = max(frac, min_area_pct if frac >= min_area_pct else 0.0)
        self.prev_gray = gray
        return float(score), mask
