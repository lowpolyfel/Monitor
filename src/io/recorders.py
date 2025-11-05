import os
import cv2
from datetime import datetime

def ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def make_videowriter(out_path, fps_in, size):
    if not out_path:
        return None
    ensure_dir(out_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 30.0 if (not fps_in) or (fps_in <= 0) else fps_in
    return cv2.VideoWriter(out_path, fourcc, fps, size)

class CSVRecorder:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        ensure_dir(csv_path)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("archivo,src,inicio_iso,fin_iso,duracion_seg\n")

    def append_interval(self, src, ts_start, ts_end):
        base = os.path.basename(str(src)) if isinstance(src, str) else f"cam_{src}"
        start_iso = datetime.fromtimestamp(ts_start).isoformat(timespec="seconds")
        end_iso   = datetime.fromtimestamp(ts_end).isoformat(timespec="seconds")
        dur = ts_end - ts_start
        with open(self.csv_path, "a", encoding="utf-8") as f:
            f.write(f"{base},{src},{start_iso},{end_iso},{dur:.3f}\n")
