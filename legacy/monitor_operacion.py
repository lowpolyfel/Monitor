#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modo continuo:
- Detecta inicio de operación al ver movimiento sostenido.
- Mantiene estado EN OPERACIÓN hasta que la inactividad (estático) alcance >= --idle_stop_sec (20 s por defecto).
- Registra un único intervalo (inicio y fin). Si el video termina y seguía en operación, cierra con la hora de fin.

Técnica CV principal: diferencia de frames (frame differencing) con umbral binario,
suavizado gaussiano, morfología (OPEN) y media móvil + umbrales con histéresis.
"""

import cv2
import numpy as np
import argparse
import time
import os
from collections import deque
from datetime import datetime

# ---------- Utils ----------
def moving_avg(vals, k):
    if not vals:
        return 0.0
    if len(vals) < k:
        return float(np.mean(vals))
    return float(np.mean(list(vals)[-k:]))

def put_text(img, text, org, scale=0.8, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thick+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def banner(img, status, score, up_thr, down_thr, fps_est):
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

def mini_graph(img, values, title="score"):
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

def iso(ts=None):
    return (datetime.now() if ts is None else datetime.fromtimestamp(ts)).isoformat(timespec="seconds")

# ---------- Args ----------
ap = argparse.ArgumentParser(description="Monitor continuo de operación (único intervalo inicio/fin).")
ap.add_argument("--src", required=True, help="Ruta de video o índice de cámara (ej. 0)")
ap.add_argument("--width", type=int, default=960, help="Ancho de procesamiento (auto alto)")
ap.add_argument("--skip", type=int, default=1, help="Procesar cada N frames")
ap.add_argument("--blur", type=int, default=5, help="Kernel gaussiano (impar)")
ap.add_argument("--diff_thr", type=int, default=18, help="Umbral de diferencia [0-255]")
ap.add_argument("--min_area_pct", type=float, default=0.002, help="Fracción mínima de pixeles activos")
ap.add_argument("--smooth_k", type=int, default=8, help="Ventana de media móvil del score")
ap.add_argument("--up_threshold", type=float, default=0.015, help="Umbral para entrar en operación")
ap.add_argument("--down_threshold", type=float, default=0.010, help="Umbral para considerar inactividad")
ap.add_argument("--min_active_sec", type=float, default=0.5, help="Antirebote para activar operación")
ap.add_argument("--idle_stop_sec", type=float, default=20.0, help="Segundos estático para fin de operación")
ap.add_argument("--out", default="", help="MP4 anotado (opcional)")
ap.add_argument("--csv", default="operacion_sesion.csv", help="CSV para guardar el único intervalo")
args = ap.parse_args()

# ---------- Fuente ----------
source = int(args.src) if args.src.isdigit() else args.src
cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise RuntimeError("No pude abrir la fuente de video/cámara.")

orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
scale  = args.width / float(orig_w) if orig_w > 0 else 1.0
proc_w = args.width if orig_w > 0 else 960
proc_h = int(orig_h * scale) if orig_h > 0 else 540

# Writer opcional
writer = None
if args.out:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, 30.0 if fps_in <= 0 else fps_in, (proc_w, proc_h))

# Primer frame
ret, frame0 = cap.read()
if not ret:
    raise RuntimeError("No pude leer el primer frame.")
frame0 = cv2.resize(frame0, (proc_w, proc_h))
if args.blur % 2 == 0:
    args.blur += 1

prev = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
prev = cv2.GaussianBlur(prev, (args.blur, args.blur), 0)

# Estado continuo
status = "IDLE"                 # "IDLE" | "OPERACION"
score_hist = deque(maxlen=max(10, args.smooth_k))
last_change = time.time()
fps_est, frames_cnt, t_fps = 0.0, 0, time.time()

# Un único intervalo
inicio_op = None
fin_op = None
intervalo_guardado = False      # aseguramos un solo registro

# CSV header si no existe
if args.csv and not os.path.exists(args.csv):
    with open(args.csv, "w", encoding="utf-8") as f:
        f.write("archivo,src,inicio_iso,fin_iso,duracion_seg\n")

print("[OK] Modo continuo iniciado. ESC para salir.")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % args.skip != 0:
        frame_idx += 1
        continue

    frame = cv2.resize(frame, (proc_w, proc_h))
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (args.blur, args.blur), 0)

    # Diferencia y máscara
    diff = cv2.absdiff(gray, prev)
    _, mask = cv2.threshold(diff, args.diff_thr, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    active = float(np.count_nonzero(mask))
    total  = float(mask.size)
    frac   = active / max(1.0, total)
    score  = max(frac, args.min_area_pct if frac >= args.min_area_pct else 0.0)
    score_hist.append(score)
    score_s = moving_avg(score_hist, args.smooth_k)

    now = time.time()
    frames_cnt += 1
    if now - t_fps >= 0.5:
        fps_est = frames_cnt / (now - t_fps)
        frames_cnt = 0
        t_fps = now

    # --------- Lógica continua ----------
    if not intervalo_guardado:
        if status == "IDLE":
            # Activar operación si el score supera up_threshold durante min_active_sec
            if score_s >= args.up_threshold and (now - last_change) >= args.min_active_sec:
                status = "OPERACION"
                last_change = now
                if inicio_op is None:
                    inicio_op = now
                # print(f"[{iso(inicio_op)}] INICIO OPERACION")
        else:  # OPERACION
            # Si cae por debajo de down_threshold, inicia contador de inactividad
            if score_s <= args.down_threshold:
                # ¿Ya alcanzó inactividad continua suficiente?
                if (now - last_change) >= args.idle_stop_sec:
                    status = "IDLE"
                    last_change = now
                    if fin_op is None:
                        fin_op = now
                        # Guardar único intervalo
                        if args.csv:
                            src_str = str(args.src)
                            base = os.path.basename(src_str) if isinstance(src_str, str) else f"cam_{src_str}"
                            with open(args.csv, "a", encoding="utf-8") as f:
                                f.write(f"{base},{src_str},{iso(inicio_op)},{iso(fin_op)},{fin_op-inicio_op:.3f}\n")
                        print(f"[OK] Fin de operación por inactividad: {iso(inicio_op)} -> {iso(fin_op)}  ({fin_op-inicio_op:.1f}s)")
                        intervalo_guardado = True
            else:
                # Sigue habiendo movimiento; refresca marca de cambio (evita sumar idle)
                last_change = now
    # ------------------------------------

    # Overlays
    out = frame.copy()
    banner(out, status, score_s, args.up_threshold, args.down_threshold, fps_est)
    mini_graph(out, list(score_hist), "score")
    mini = cv2.resize(mask, (0,0), fx=0.25, fy=0.25)
    mini = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)
    hm, wm = mini.shape[:2]
    out[-hm-10:-10, 10:10+wm] = mini
    put_text(out, "mask", (12, out.shape[0]-hm-14), 0.5, (200,200,200), 1)

    cv2.imshow("Monitor continuo", out)
    if writer is not None:
        writer.write(out)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    prev = gray
    frame_idx += 1

# Si el video termina y la operación estaba activa y no se guardó intervalo aún, cerramos con el fin del video
if (inicio_op is not None) and (not intervalo_guardado):
    fin_op = time.time()
    if args.csv:
        src_str = str(args.src)
        base = os.path.basename(src_str) if isinstance(src_str, str) else f"cam_{src_str}"
        with open(args.csv, "a", encoding="utf-8") as f:
            f.write(f"{base},{src_str},{iso(inicio_op)},{iso(fin_op)},{fin_op-inicio_op:.3f}\n")
    print(f"[OK] Fin de operación por fin de fuente: {iso(inicio_op)} -> {iso(fin_op)}  ({fin_op-inicio_op:.1f}s)")

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
print("[OK] Finalizado.")

