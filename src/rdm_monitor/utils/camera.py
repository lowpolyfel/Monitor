cat > src/rdm_monitor/utils/camera.py <<'PY'
import cv2
import threading
import time
from collections import deque

class FrameReader:
    """
    Lector en hilo: mantiene solo el frame más reciente (cola=1).
    Alivia latencia de VideoCapture.read() y sube FPS efectivos.
    """
    def __init__(self, cap, maxlen=1):
        self.cap = cap
        self.queue = deque(maxlen=maxlen)
        self.stop_flag = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.t.start()
        return self

    def _loop(self):
        while not self.stop_flag.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            self.queue.append(frame)

    def read_latest(self):
        if self.queue:
            return True, self.queue[-1]
        return False, None

    def stop(self):
        self.stop_flag.set()
        self.t.join(timeout=1.0)
PY

