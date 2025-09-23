import os
import sys
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

RAW_DIR = "data/raw_videos"

def list_videos():
    if not os.path.isdir(RAW_DIR):
        return []
    exts = (".mp4", ".mov", ".mkv", ".avi", ".MP4", ".MOV", ".MKV", ".AVI")
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(exts)]
    return sorted(files)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("rdm-monitor | Selección de fuente")
        self.geometry("520x560")  # ventana “normal”
        self.resizable(False, False)

        # Estado
        self.source_mode = tk.StringVar(value="camera")  # 'camera' | 'video'
        self.camera_index = tk.StringVar(value="0")
        # Ahora con 6 motores
        self.engine = tk.StringVar(value="mog2")         # 'mog2'|'diff'|'knn'|'flow'|'avg'|'edges'
        self.idle_stop = tk.StringVar(value="20")
        self.width = tk.StringVar(value="960")           # procesamiento
        self.display_width = tk.StringVar(value="960")   # ventana OpenCV
        self.up_thr = tk.StringVar(value="0.008")
        self.down_thr = tk.StringVar(value="0.005")
        self.min_area = tk.StringVar(value="0.001")
        self.smooth_k = tk.StringVar(value="8")
        self.skip = tk.StringVar(value="1")
        self.out_video = tk.BooleanVar(value=True)
        self.debug = tk.BooleanVar(value=False)

        self._build_ui()
        self._refresh_videos()

    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # Fuente
        frm_src = ttk.LabelFrame(self, text="Fuente")
        frm_src.pack(fill="x", **pad)

        rb1 = ttk.Radiobutton(frm_src, text="Cámara USB", variable=self.source_mode, value="camera", command=self._toggle_source)
        rb2 = ttk.Radiobutton(frm_src, text="Video en data/raw_videos", variable=self.source_mode, value="video", command=self._toggle_source)
        rb1.grid(row=0, column=0, sticky="w", padx=6, pady=4)
        rb2.grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(frm_src, text="Índice cámara:").grid(row=1, column=0, sticky="e")
        self.ent_cam = ttk.Entry(frm_src, width=6, textvariable=self.camera_index)
        self.ent_cam.grid(row=1, column=1, sticky="w")

        # Lista de videos
        frm_vid = ttk.Frame(frm_src)
        frm_vid.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6,0))
        frm_vid.columnconfigure(0, weight=1)

        self.lst = tk.Listbox(frm_vid, height=8, exportselection=False)
        self.lst.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(frm_vid, orient="vertical", command=self.lst.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.lst.configure(yscrollcommand=sb.set)

        btn_refresh = ttk.Button(frm_vid, text="Refrescar lista", command=self._refresh_videos)
        btn_refresh.grid(row=1, column=0, sticky="w", pady=6)

        # Parámetros
        frm_params = ttk.LabelFrame(self, text="Parámetros")
        frm_params.pack(fill="x", **pad)

        row = 0
        ttk.Label(frm_params, text="Motor:").grid(row=row, column=0, sticky="e")
        ttk.Combobox(
            frm_params,
            values=["mog2","diff","knn","flow","avg","edges"],
            textvariable=self.engine,
            width=10,
            state="readonly"
        ).grid(row=row, column=1, sticky="w")

        ttk.Checkbutton(frm_params, text="Guardar video anotado", variable=self.out_video).grid(row=row, column=2, columnspan=2, sticky="w")
        row += 1

        ttk.Label(frm_params, text="idle_stop_sec:").grid(row=row, column=0, sticky="e")
        ttk.Entry(frm_params, width=6, textvariable=self.idle_stop).grid(row=row, column=1, sticky="w")
        ttk.Label(frm_params, text="display_width:").grid(row=row, column=2, sticky="e")
        ttk.Entry(frm_params, width=6, textvariable=self.display_width).grid(row=row, column=3, sticky="w")
        row += 1

        ttk.Label(frm_params, text="width(proc):").grid(row=row, column=0, sticky="e")
        ttk.Entry(frm_params, width=6, textvariable=self.width).grid(row=row, column=1, sticky="w")
        ttk.Label(frm_params, text="skip:").grid(row=row, column=2, sticky="e")
        ttk.Entry(frm_params, width=6, textvariable=self.skip).grid(row=row, column=3, sticky="w")
        row += 1

        ttk.Label(frm_params, text="up_thr:").grid(row=row, column=0, sticky="e")
        ttk.Entry(frm_params, width=6, textvariable=self.up_thr).grid(row=row, column=1, sticky="w")
        ttk.Label(frm_params, text="down_thr:").grid(row=row, column=2, sticky="e")
        ttk.Entry(frm_params, width=6, textvariable=self.down_thr).grid(row=row, column=3, sticky="w")
        row += 1

        ttk.Label(frm_params, text="min_area_pct:").grid(row=row, column=0, sticky="e")
        ttk.Entry(frm_params, width=6, textvariable=self.min_area).grid(row=row, column=1, sticky="w")
        ttk.Label(frm_params, text="smooth_k:").grid(row=row, column=2, sticky="e")
        ttk.Entry(frm_params, width=6, textvariable=self.smooth_k).grid(row=row, column=3, sticky="w")
        row += 1

        ttk.Checkbutton(frm_params, text="Debug (imprime score)", variable=self.debug).grid(row=row, column=0, columnspan=4, sticky="w")
        row += 1

        # Acciones
        frm_actions = ttk.Frame(self)
        frm_actions.pack(fill="x", **pad)
        ttk.Button(frm_actions, text="Iniciar monitor", command=self._start).pack(side="left")
        ttk.Button(frm_actions, text="Salir", command=self.destroy).pack(side="right")

        self._toggle_source()

    def _toggle_source(self):
        mode = self.source_mode.get()
        state_cam = "normal" if mode=="camera" else "disabled"
        state_vid = "normal" if mode=="video" else "disabled"
        self.ent_cam.configure(state=state_cam)
        self.lst.configure(state=state_vid)

    def _refresh_videos(self):
        self.lst.delete(0, tk.END)
        vids = list_videos()
        if not vids:
            self.lst.insert(tk.END, "(Sin videos en data/raw_videos)")
            return
        for v in vids:
            self.lst.insert(tk.END, v)

    def _selected_video_path(self):
        if not self.lst.size():
            return None
        sel = self.lst.curselection()
        if not sel:
            return None
        basename = self.lst.get(sel[0])
        if basename.startswith("(Sin videos"):
            return None
        return os.path.join(RAW_DIR, basename)

    def _start(self):
        # Construir comando a ejecutar (subproceso)
        if self.source_mode.get() == "camera":
            src = self.camera_index.get().strip() or "0"
        else:
            path = self._selected_video_path()
            if not path:
                messagebox.showerror("Error", "Selecciona un video de la lista.")
                return
            src = path

        engine = self.engine.get()
        idle_stop = self.idle_stop.get()
        width = self.width.get()
        disp_w = self.display_width.get()
        up = self.up_thr.get()
        down = self.down_thr.get()
        min_area = self.min_area.get()
        smooth_k = self.smooth_k.get()
        skip = self.skip.get()
        debug = self.debug.get()

        # Salida anotada opcional
        out_arg = []
        if self.out_video.get():
            if src.isdigit():
                out_name = "cam_{}_annotated.mp4".format(src)
            else:
                base = os.path.splitext(os.path.basename(src))[0]
                out_name = f"{base}_annotated.mp4"
            out_arg = ["--out", os.path.join("outputs", "annotated", out_name)]

        cmd = [
            sys.executable, "-m", "rdm_monitor",
            "--engine", engine,
            "--src", src,
            "--idle_stop_sec", idle_stop,
            "--width", width,
            "--display_width", disp_w,
            "--up_threshold", up,
            "--down_threshold", down,
            "--min_area_pct", min_area,
            "--smooth_k", smooth_k,
            "--skip", skip,
            "--csv", os.path.join("outputs", "csv", "operacion_sesion.csv"),
        ] + out_arg

        if debug:
            cmd.append("--debug")

        env = os.environ.copy()
        env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

        try:
            self.destroy()
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Falló la ejecución:\n{e}")

def main():
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()
