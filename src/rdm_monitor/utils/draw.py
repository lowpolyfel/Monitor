import cv2
import numpy as np

def put_text(img, text, org, scale=0.8, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_status_banner(img, status, score, up_thr, down_thr, fps_est):
    h, w = img.shape[:2]
    bar_h = 48
    overlay = img.copy()
    color = (0,170,0) if status == "OPERACION" else (60,60,200)
    cv2.rectangle(overlay, (0,0), (w, bar_h), color, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    put_text(img, f"Estado: {'EN OPERACION' if status=='OPERACION' else 'IDLE'}",
             (10, 32), 0.8, (255,255,255), 2)
    put_text(img, f"Score:{score:.3f} Up:{up_thr:.3f} Down:{down_thr:.3f} FPS:{fps_est:.1f}",
             (10, bar_h+26), 0.6, (255,255,0), 2)

def draw_small_graph(img, values, title="score"):
    gh, gw = 64, 240
    x0, y0 = img.shape[1]-gw-10, 10
    g = np.zeros((gh, gw, 3), np.uint8)
    if len(values) > 1:
        v = np.clip(np.array(values, dtype=np.float32), 0, 1.0)
        xs = np.linspace(8, gw-8, len(v)).astype(int)
        ys = (1.0 - v) * (gh-12) + 6
        pts = np.stack([xs, ys.astype(int)], axis=1)
        for i in range(1, len(pts)):
            cv2.line(g, tuple(pts[i-1]), tuple(pts[i]), (0,255,0), 2)
    cv2.rectangle(g, (0,0), (gw-1, gh-1), (100,100,100), 1)
    put_text(g, title, (6,18), 0.6, (200,200,200), 1)
    img[y0:y0+gh, x0:x0+gw] = g

def draw_mini_mask(img, mask, bottom_left, scale=0.25):
    mini = cv2.resize(mask, (0,0), fx=scale, fy=scale)
    mini = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)
    hm, wm = mini.shape[:2]
    x = bottom_left[0]
    y = bottom_left[1] - hm
    img[y:y+hm, x:x+wm] = mini
    put_text(img, "mask", (x+2, y-4), 0.5, (200,200,200), 1)
