"""Utility helpers for working with OpenCV ``VideoCapture`` sources."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple

import cv2


class FrameReader:
    """Continuously pull frames from a ``cv2.VideoCapture`` in a background thread.

    ``VideoCapture.read`` is a blocking call and, depending on the backend,
    introduces noticeable latency. ``FrameReader`` mitigates this by polling
    the capture on a separate daemon thread and keeping only the most recent
    frame in a bounded deque.  The consumer can then call :meth:`read_latest`
    to obtain the freshest frame available without waiting for I/O.
    """

    def __init__(self, cap: cv2.VideoCapture, maxlen: int = 1) -> None:
        self.cap = cap
        self.queue: Deque = deque(maxlen=maxlen)
        self.stop_flag = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> "FrameReader":
        """Start polling frames in the background."""
        if not self.thread.is_alive():
            self.thread.start()
        return self

    def _loop(self) -> None:
        while not self.stop_flag.is_set():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.005)
                continue
            self.queue.append(frame)

    def read_latest(self) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """Return the most recently captured frame if available."""
        if self.queue:
            return True, self.queue[-1]
        return False, None

    def stop(self, timeout: float = 1.0) -> None:
        """Stop the background thread and wait for it to finish."""
        self.stop_flag.set()
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)
