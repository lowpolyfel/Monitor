import cv2
import numpy as np
import argparse
from collections import deque
import time
import os

# ---------- Utils ----------
def moving_avg(x, k=5):
    if len(x) < k: 
        return x[-1] if x else 0.0
    return float(np.mean(list(x)[-k:]))

def put_text(img, text, org, scale=0.7, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

# ---------- Args ----------
ap = argparse.ArgumentParser(description="Demo: conteo de ciclos por movimiento de la punta")
ap.add_argument("--src", required=True, help="Ruta de video o índice de cámara (por ej. 0)")
ap.add_argument("--out", default="output_annotated.mp4", help="MP4 de salida")
ap.add_argument("--width", type=int, default=960, help="ancho de procesamiento (auto alto)")
ap.add_argument("--min_period_ms", type=int, default=250, help="anti-rebote: mínimo intervalo entre ciclos")
ap.add_argument("--mag_threshold", type=float, default=1.6, help="umbral de magnitud media (pix/frame)")
ap.add_argument("--skip", type=int, default=1, help="procesar cada N frames")
args = ap.parse_args()

# ---------- Fuente ----------
source = 0 if (args.src.isdigit()) else args.src
cap = cv2.VideoCapture(int(source) if isinstance(source,int) else source)
if not cap.isOpened():
    raise RuntimeError("No pude abrir la fuente de video.")

# Dimensiones y writer
orig_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30.0
scale   = args.width / float(orig_w)
proc_w  = args.width
proc_h  = int(orig_h * scale)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(args.out, fourcc, fps_in, (proc_w, proc_h))

# ---------- Selección de ROI inicial (punta) ----------
ret, frame0 = cap.read()
if not ret: 
    raise RuntimeError("No pude leer el primer frame.")
frame0 = cv2.resize(frame0, (proc_w, proc_h))
disp = frame0.copy()
put_text(disp, "Dibuja un ROI pequeno sobre la PUNTA y presiona ENTER.", (20,30))
r = cv2.selectROI("Seleccion ROI (punta)", disp, False, False)
cv2.destroyWindow("Seleccion ROI (punta)")
if r[2] == 0 or r[3] == 0:
    raise RuntimeError("ROI invalido.")

x,y,w,h = map(int, r)

# Tracker para acompañar el ROI (CSRT es robusto para objetos pequenos)
try:
    tracker = cv2.legacy.TrackerCSRT_create()
except:
    tracker = cv2.TrackerCSRT_create()
tracker.init(frame0, (x,y,w,h))

# Puntos para flujo óptico
gray0 = cv2.cvtColor(frame0[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
feat_params = dict(maxCorners=60, qualityLevel=0.01, minDistance=4, blockSize=7)
p0 = cv2.goodFeaturesToTrack(gray0, mask=None, **feat_params)
if p0 is None:
    # fallback: puntos en rejilla
    yy, xx = np.mgrid[4:h-4:8, 4:w-4:8]
    p0 = np.vstack((xx.ravel(), yy.ravel())).T.astype(np.float32).reshape(-1,1,2)

lk_params = dict(winSize=(13,13), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

prev_gray_full = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
prev_gray_roi  = prev_gray_full[y:y+h, x:x+w].copy()
prev_pts       = p0.copy()

# Buffers y conteo
mag_hist = deque(maxlen=15)
vel_hist = deque(maxlen=15)
count    = 0
last_event_ts = 0.0

# Línea horizonte de magnitud para dibujo
def draw_hud(img, count, mag_avg, thr):
    put_text(img, f"Ciclos: {count}", (20,40), 1.0, (0,255,0), 2)
    put_text(img, f"Mag(avg): {mag_avg:.2f}  Thr:{thr:.2f}", (20,70), 0.7, (255,255,0), 2)

# ---------- Bucle principal ----------
frame_idx = 0
t0 = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break
    if frame_idx % args.skip != 0:
        frame_idx += 1
        continue

    frame = cv2.resize(frame, (proc_w, proc_h))
    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Actualizar tracker para seguir el ROI de la punta
    ok, bbox = tracker.update(frame)
    if ok:
        x,y,w,h = [int(v) for v in bbox]
        x = max(0, min(x, proc_w-1))
        y = max(0, min(y, proc_h-1))
        w = max(10, min(w, proc_w - x))
        h = max(10, min(h, proc_h - y))
    roi = frame[y:y+h, x:x+w]
    gray_roi = gray_full[y:y+h, x:x+w]

    # Recalcular puntos si se pierden muchos
    if prev_pts is None or len(prev_pts) < 10:
        prev_pts = cv2.goodFeaturesToTrack(gray_roi, mask=None, **feat_params)
        if prev_pts is None:
            yy, xx = np.mgrid[4:h-4:8, 4:w-4:8]
            prev_pts = np.vstack((xx.ravel(), yy.ravel())).T.astype(np.float32).reshape(-1,1,2)
        prev_gray_roi = gray_roi.copy()

    # Flujo óptico Lucas-Kanade dentro del ROI
    p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray_roi, gray_roi, prev_pts, None, **lk_params)
    if p1 is not None and prev_pts is not None:
        good_new = p1[st==1]
        good_old = prev_pts[st==1]

        # Vector promedio de movimiento (pix/frame)
        if len(good_new) > 0:
            diff = (good_new - good_old)
            vx = float(np.median(diff[:,0]))
            vy = float(np.median(diff[:,1]))
            vmag = float(np.hypot(vx, vy))
            vel_hist.append(vy)            # puedes usar eje Y si el movimiento es vertical
            mag_hist.append(vmag)
        else:
            vx = vy = vmag = 0.0
    else:
        vx = vy = vmag = 0.0

    # Suavizado y detección de pico (evento)
    mag_avg = moving_avg(mag_hist, k=5)
    now = time.time()
    min_period = args.min_period_ms / 1000.0

    # Condición de ciclo: magnitud supera umbral y ha pasado el periodo mínimo
    if mag_avg > args.mag_threshold and (now - last_event_ts) > min_period:
        count += 1
        last_event_ts = now

    # Dibujo
    out = frame.copy()
    # ROI
    cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,255), 2)
    put_text(out, "ROI punta", (x, max(0,y-8)), 0.6, (0,255,255), 2)

    # Vectores de flujo (muestra)
    if p1 is not None and prev_pts is not None and st is not None:
        for (n, o) in zip(p1[st==1], prev_pts[st==1]):
            a,b = n.ravel()
            c,d = o.ravel()
            cv2.line(out, (x+int(a), y+int(b)), (x+int(c), y+int(d)), (255,0,0), 1)
            cv2.circle(out, (x+int(a), y+int(b)), 2, (0,0,255), -1)

    draw_hud(out, count, mag_avg, args.mag_threshold)

    # Pequeño gráfico de magnitud
    graph_h, graph_w = 60, 220
    graph = np.zeros((graph_h, graph_w, 3), np.uint8)
    vals = np.array([min(10.0,v) for v in mag_hist], dtype=np.float32)
    if len(vals)>1:
        vals = (1.0 - vals/10.0) * (graph_h-10)
        xs = np.linspace(10, graph_w-10, len(vals)).astype(int)
        pts = np.stack([xs, vals.astype(int)], axis=1)
        for i in range(1, len(pts)):
            cv2.line(graph, tuple(pts[i-1]), tuple(pts[i]), (0,255,0), 2)
    cv2.rectangle(graph, (0,0), (graph_w-1, graph_h-1), (80,80,80), 1)
    put_text(graph, "mag", (6,16), 0.5, (200,200,200), 1)
    out[10:10+graph_h, proc_w-10-graph_w:proc_w-10] = graph

    cv2.imshow("Bond Counter Demo", out)
    writer.write(out)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

    # Update para el siguiente paso
    prev_gray_roi = gray_roi.copy()
    prev_pts = p1

    frame_idx += 1

cap.release()
writer.release()
cv2.destroyAllWindows()

print(f"[OK] Guardado: {os.path.abspath(args.out)} | Ciclos contados: {count}")
