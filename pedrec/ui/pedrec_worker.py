import threading
import time
from typing import Optional

import numpy as np
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import QApplication

from pedrec.utils.input_providers.input_provider_base import InputProviderBase


class PedRecWorker(QThread):
    """
    Processing thread of the GUI. Frames are handed to the GUI one at a time: the next frame is only emitted after the
    GUI has drawn the previous one (``frame_consumed``). Without this backpressure the queued frames pile up whenever
    drawing is slower than the processing, and the display stalls.
    """
    data_updated = Signal(int, np.ndarray, list, np.ndarray, int)
    input_provider: InputProviderBase = None

    def __init__(self, parent: QApplication, max_fps: Optional[float] = None):
        """:param max_fps: limit the playback speed (e.g. to the frame rate of a video file), None = unlimited"""
        QThread.__init__(self, parent)
        self.window = parent
        self._lock = threading.Lock()
        self._frame_slot = threading.Semaphore(1)
        self.min_frame_interval = 1.0 / max_fps if max_fps else 0.0
        self._last_emit = 0.0
        self.active = True
        self.paused = False

    def stop(self):
        """Stops the processing loop and waits for the thread (Qt aborts if a running QThread is destroyed)."""
        with self._lock:
            self.active = False
        self._frame_slot.release()
        if hasattr(self.input_provider, "stop"):
            self.input_provider.stop()
        self.wait(10000)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def frame_consumed(self):
        """Called by the GUI after it has taken over the last emitted frame."""
        self._frame_slot.release()

    def emit_frame(self, *data):
        """Waits until the GUI is ready (and for the playback speed limit), then emits the frame."""
        while not self._frame_slot.acquire(timeout=0.1):
            with self._lock:
                if not self.active:
                    return
        wait = self._last_emit + self.min_frame_interval - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
        self._last_emit = time.perf_counter()
        self.data_updated.emit(*data)

    def run(self):
        if self.input_provider is None:
            raise ValueError("No input provider set")
        frame_nr = 0
        iterator = self.input_provider.get_data()
        while True:
            with self._lock:
                if not self.active:
                    return
            if self.paused:
                QThread.msleep(100)
                continue
            img = next(iterator, None)
            if img is None:
                return
            frame_nr += 1
            self.run_impl(frame_nr, img)

    def run_impl(self, frame_nr: int, img: np.ndarray):
        raise NotImplementedError
