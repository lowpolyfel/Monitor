import cv2
import numpy as np
import argparse
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from .utils.draw import put_text, draw_status_banner, draw_small_graph, draw_mini_mask
from .utils.smoothing import moving_avg
from .pipeline.diff_motion import DiffMotion
from .pipeline.mog2_motion import Mog2Motion
from .pipeline.knn_motion import KNNMotion
from .pipeline.flow_motion import FlowMotion
from .pipeline.avg_motion import AvgMotion
from .pipeline.edges_motion import EdgesMotion
from .io.recorders import CSVRecorder, make_videowriter

def iso(ts=None):
    return (datetime.now() if ts is None else datetime.fromtimestamp(ts)).isoformat(timespec="seconds")

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV", ".MKV", ".AVI")
DEFAULT_VIDEO_DIR = "data/raw_videos"


def list_videos(raw_dir: str) -> list[str]:
    """Enumerate video files inside *raw_dir* sorted alphabetically."""

    if not os.path.isdir(raw_dir):
        return []

    entries = []
    with os.scandir(raw_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(VIDEO_EXTENSIONS):
                entries.append(entry.path)
    return sorted(entries)


def choose_video_from_directory(directory: str) -> str | None:
    """Prompt the user to pick a video file from ``directory``."""

    videos = list_videos(directory)
    if not videos:
        print(f"No se encontraron videos en {directory}.")
        return None

    print(f"\nVideos disponibles en {directory}:")
    for index, path in enumerate(videos, 1):
        print(f"{index}) {os.path.basename(path)}")

    selection = input("Elige número: ").strip()
    if not selection.isdigit():
        print("Selección inválida.")
        return None

    idx = int(selection)
    if idx < 1 or idx > len(videos):
        print("Selección fuera de rango.")
        return None
    return videos[idx - 1]


def prompt_video_directory(default: str = DEFAULT_VIDEO_DIR) -> str | None:
    """Ask the user for a directory that contains videos to analyse."""

    answer = input(f"Directorio de videos [{default}]: ").strip()
    chosen = Path(answer) if answer else Path(default)
    if not chosen.exists() or not chosen.is_dir():
        print("El directorio indicado no existe.")
        return None
    return str(chosen)


def pick_source_interactive() -> str | None:
    print("\n== Monitor: Selecciona fuente ==")
    print("1) Cámara USB (índice 0)")
    print("2) Seleccionar video desde un directorio")
    print("3) Especificar ruta manual")
    choice = input("Opción [1/2/3]: ").strip() or "1"
    if choice == "1":
        return "0"
    if choice == "2":
        directory = prompt_video_directory()
        if not directory:
            return None
        return choose_video_from_directory(directory)
    if choice == "3":
        path = input("Ruta al video/cámara (ej. 0 o data/raw_videos/mi.mp4): ").strip()
        return path if path else None

    print("Opción inválida.")
    return None

def make_engine(name, args):
    if name == "diff":
        blur = args.blur if (args.blur % 2 == 1) else args.blur + 1
        return DiffMotion(blur=blur, diff_thr=args.diff_thr,
                          min_blob_area_px=args.min_blob_area_px,
                          morph_kernel=args.morph_kernel)
    if name == "mog2":
        return Mog2Motion(history=args.mog2_history, varThreshold=args.mog2_varT,
                          detectShadows=True, learningRate=args.mog2_lr,
                          min_blob_area_px=args.min_blob_area_px,
                          morph_kernel=args.morph_kernel)
    if name == "knn":
        return KNNMotion(history=args.knn_history, dist2Threshold=args.knn_dist2T,
                         detectShadows=True, learningRate=args.knn_lr,
                         min_blob_area_px=args.min_blob_area_px,
                         morph_kernel=args.morph_kernel)
    if name == "flow":
        return FlowMotion(flow_thr=args.flow_thr,
                          min_blob_area_px=args.min_blob_area_px,
                          morph_kernel=args.morph_kernel)
    if name == "avg":
        return AvgMotion(alpha=args.avg_alpha, thr=args.avg_thr,
                         min_blob_area_px=args.min_blob_area_px,
                         morph_kernel=args.morph_kernel)
    if name == "edges":
        return EdgesMotion(canny1=args.canny1, canny2=args.canny2,
                           min_blob_area_px=args.min_blob_area_px,
                           morph_kernel=args.morph_kernel)
    raise ValueError(f"Engine desconocido: {name}")

def main():
    ap = argparse.ArgumentParser(
        description="Monitor de operación con múltiples motores y múltiples intervalos (OPERACION <-> IDLE)."
    )
    ap.add_argument(
        "--src",
        default="",
        help="Ruta de video, índice de cámara o directorio de videos. Si se omite, aparece un menú.",
    )
    ap.add_argument("--engine", choices=["mog2","diff","knn","flow","avg","edges"], default="mog2",
                    help="Motor de movimiento.")
    # Dimensiones
    ap.add_argument("--width", type=int, default=960, help="Ancho de procesamiento (auto alto)")
    ap.add_argument("--display_width", type=int, default=960, help="Ancho de ventana OpenCV")
    ap.add_argument("--skip", type=int, default=1, help="Procesar cada N frames")
    # Umbrales generales
    ap.add_argument("--min_area_pct", type=float, default=0.001, help="Frac. mínima de pixeles activos (0-1)")
    ap.add_argument("--smooth_k", type=int, default=8, help="Ventana de media móvil del score")
    ap.add_argument("--up_threshold", type=float, default=0.008, help="Umbral para entrar en operación")
    ap.add_argument("--down_threshold", type=float, default=0.005, help="Umbral para considerar inactividad")
    ap.add_argument("--min_active_sec", type=float, default=0.5, help="Antirebote para activar operación")
    ap.add_argument("--idle_stop_sec", type=float, default=20.0, help="Segundos estático para cerrar intervalo")
    ap.add_argument("--out", default="", help="MP4 anotado (opcional)")
    ap.add_argument("--csv", default="outputs/csv/operacion_sesion.csv", help="CSV de intervalos")
    ap.add_argument("--debug", action="store_true", help="Imprime score/estado periódico")

    # Afinado MOG2 / KNN / Diff / Flow / Avg / Edges
    ap.add_argument("--mog2_history", type=int, default=600)
    ap.add_argument("--mog2_varT", type=float, default=25.0)
    ap.add_argument("--mog2_lr", type=float, default=-1.0, help="-1 = auto")

    ap.add_argument("--knn_history", type=int, default=600)
    ap.add_argument("--knn_dist2T", type=float, default=400.0)
    ap.add_argument("--knn_lr", type=float, default=-1.0)

    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--diff_thr", type=int, default=18)

    ap.add_argument("--flow_thr", type=float, default=0.7, help="Umbral de magnitud de flujo (px/frame)")

    ap.add_argument("--avg_alpha", type=float, default=0.02, help="Factor de aprendizaje promedio exponencial")
    ap.add_argument("--avg_thr", type=int, default=18)

    ap.add_argument("--canny1", type=int, default=60)
    ap.add_argument("--canny2", type=int, default=120)

    ap.add_argument("--morph_kernel", type=int, default=3, help="Kernel morfológico (3/5)")
    ap.add_argument("--min_blob_area_px", type=int, default=80, help="Descartar blobs pequeños (px)")

    args = ap.parse_args()

    # Menú si no se especificó --src
    if not args.src:
        src_pick = pick_source_interactive()
        if not src_pick:
            print("No se seleccionó fuente. Saliendo.")
            return
        args.src = src_pick

    if os.path.isdir(args.src):
        chosen = choose_video_from_directory(args.src)
        if not chosen:
            print("No se seleccionó un video dentro del directorio. Saliendo.")
            return
        args.src = chosen

    # Fuente
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

    writer = make_videowriter(args.out, fps_in, (proc_w, proc_h)) if args.out else None

    # Motor
    engine = make_engine(args.engine, args)

    # Primer frame / init
    ret, frame0 = cap.read()
    if not ret:
        raise RuntimeError("No pude leer el primer frame.")
    frame0 = cv2.resize(frame0, (proc_w, proc_h))
    engine.initialize(frame0)

    # Ventana
    cv2.namedWindow("monitor", cv2.WINDOW_NORMAL)
    disp_h = int(proc_h * (args.display_width / proc_w))
    cv2.resizeWindow("monitor", args.display_width, disp_h)

    # Estado con múltiples intervalos
    status = "IDLE"                 # "IDLE" | "OPERACION"
    score_hist = deque(maxlen=max(10, args.smooth_k))
    last_state_ts = time.time()
    fps_est, frames_cnt, t_fps = 0.0, 0, time.time()

    # Intervalos: lista de tuplas (inicio, fin)
    current_start = None
    csv_rec = CSVRecorder(args.csv) if args.csv else None

    print("[OK] Monitor iniciado. ESC para salir.")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.skip != 0:
            frame_idx += 1
            continue

        frame = cv2.resize(frame, (proc_w, proc_h))
        score, mask = engine.step(frame, min_area_pct=args.min_area_pct)

        score_hist.append(score)
        score_s = moving_avg(score_hist, args.smooth_k)

        now = time.time()
        frames_cnt += 1
        if now - t_fps >= 0.5:
            fps_est = frames_cnt / (now - t_fps)
            frames_cnt = 0
            t_fps = now

        if args.debug and frame_idx % 15 == 0:
            print(f"[DBG] score_s={score_s:.5f}  state={status}")

        # ---- Lógica con múltiples intervalos ----
        if status == "IDLE":
            if score_s >= args.up_threshold:
                # Debe sostenerse por min_active_sec
                if (now - last_state_ts) >= args.min_active_sec:
                    status = "OPERACION"
                    last_state_ts = now
                    current_start = now
                    if args.debug:
                        print(f"[STATE] -> OPERACION @ {iso(current_start)}")
            else:
                # sigue en IDLE, refrescamos el temporizador
                last_state_ts = now
        else:  # OPERACION
            if score_s <= args.down_threshold:
                # Debe sostenerse inactivo por idle_stop_sec
                if (now - last_state_ts) >= args.idle_stop_sec:
                    end_ts = now
                    if csv_rec and current_start is not None:
                        csv_rec.append_interval(args.src, current_start, end_ts)
                        print(f"[OK] Intervalo: {iso(current_start)} -> {iso(end_ts)}  ({end_ts-current_start:.1f}s)")
                    status = "IDLE"
                    last_state_ts = now
                    current_start = None
            else:
                # sigue en operacion, refrescar temporizador para idle
                last_state_ts = now
        # ----------------------------------------

        out = frame.copy()
        draw_status_banner(out, status, score_s, args.up_threshold, args.down_threshold, fps_est)
        draw_small_graph(out, list(score_hist), "score")
        if mask is not None:
            draw_mini_mask(out, mask, (10, out.shape[0]-10))

        cv2.imshow("monitor", out)
        if writer is not None:
            writer.write(out)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        frame_idx += 1

    # Cierre si quedó activo
    if status == "OPERACION" and current_start is not None:
        end_ts = time.time()
        if csv_rec:
            csv_rec.append_interval(args.src, current_start, end_ts)
        print(f"[OK] Intervalo al cierre: {iso(current_start)} -> {iso(end_ts)}  ({end_ts-current_start:.1f}s)")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print("[OK] Finalizado.")
