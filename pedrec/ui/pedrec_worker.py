import threading

import numpy as np
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import QApplication

from pedrec.utils.input_providers.input_provider_base import InputProviderBase


class PedRecWorker(QThread):
    data_updated = Signal(int, np.ndarray, list, np.ndarray, int)
    input_provider: InputProviderBase = None

    def __init__(self, parent: QApplication):
        QThread.__init__(self, parent)
        self.window = parent
        self._lock = threading.Lock()
        self.active = True
        self.paused = False

    def stop(self):
        """Stops the processing loop and waits for the thread (Qt aborts if a running QThread is destroyed)."""
        with self._lock:
            self.active = False
        if hasattr(self.input_provider, "stop"):
            self.input_provider.stop()
        self.wait(10000)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

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
