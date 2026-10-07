import queue
import threading
import time
from typing import Optional, Tuple

import numpy as np
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import QApplication


class PedRecWorker(QThread):
    """
    Processing thread of the GUI and player logic (play / pause, seeking, frame by frame).

    Commands from the GUI (``play``, ``pause``, ``seek``, ``step``, ...) are queued and executed by the thread.
    Frames are handed to the GUI one at a time: the next frame is only emitted after the GUI has drawn the previous
    one (``frame_consumed``), otherwise frames pile up whenever drawing is slower than the processing.
    """
    data_updated = Signal(int, np.ndarray, list, np.ndarray, int)
    position_changed = Signal(int, int)  # shown frame index, number of frames (0: live source / unknown)
    playing_changed = Signal(bool)

    def __init__(self, parent: QApplication, source, max_fps: Optional[float] = None):
        """
        :param source: frame source (``pedrec.ui.frame_source``)
        :param max_fps: limit the playback speed (e.g. to the frame rate of a video file), None = unlimited
        """
        QThread.__init__(self, parent)
        self.window = parent
        self.source = source
        self.min_frame_interval = 1.0 / max_fps if max_fps else 0.0
        self._lock = threading.Lock()
        self._frame_slot = threading.Semaphore(1)
        self._commands: "queue.Queue[Tuple]" = queue.Queue()
        self._last_emit = 0.0
        self.active = True
        self.playing = True
        self.current = -1  # index of the frame shown last
        self._target: Optional[int] = 0  # next frame to show

    # ------------------------------------------------------------------------------------------------ commands (GUI)
    def play(self):
        self._commands.put(("play",))

    def pause(self):
        self._commands.put(("pause",))

    def toggle_play(self):
        self._commands.put(("toggle",))

    def replay(self):
        self._commands.put(("replay",))

    def seek(self, index: int):
        self._commands.put(("seek", int(index)))

    def step(self, delta: int):
        """Pauses and shows the frame ``delta`` frames before / after the current one."""
        self._commands.put(("step", int(delta)))

    def skip_seconds(self, seconds: float):
        self._commands.put(("skip", float(seconds)))

    @property
    def seekable(self) -> bool:
        return getattr(self.source, "seekable", False)

    def stop(self):
        """Stops the processing loop and waits for the thread (Qt aborts if a running QThread is destroyed)."""
        with self._lock:
            self.active = False
        self._frame_slot.release()
        self._commands.put(("noop",))
        self.wait(10000)
        if hasattr(self.source, "stop"):
            self.source.stop()

    def frame_consumed(self):
        """Called by the GUI after it has taken over the last emitted frame."""
        self._frame_slot.release()

    # ------------------------------------------------------------------------------------------------ thread
    def _clamp(self, index: int) -> int:
        count = self.source.frame_count
        return max(0, min(index, count - 1)) if count > 0 else max(0, index)

    def _set_playing(self, playing: bool):
        if playing != self.playing:
            self.playing = playing
            self.playing_changed.emit(playing)

    def _apply(self, command: Tuple):
        name = command[0]
        base = self._target if self._target is not None else self.current
        at_end = self.source.frame_count > 0 and self.current >= self.source.frame_count - 1
        if name == "play" or (name == "toggle" and not self.playing):
            if at_end and self.seekable:
                self._target = 0
            self._set_playing(True)
        elif name == "pause" or name == "toggle":
            self._set_playing(False)
        elif name == "replay":
            self._target = 0
            self._set_playing(True)
        elif not self.seekable:
            return
        elif name == "seek":
            self._target = self._clamp(command[1])
        elif name == "step":
            self._set_playing(False)
            self._target = self._clamp(base + command[1])
        elif name == "skip":
            self._target = self._clamp(base + int(round(command[1] * self.source.fps)))

    def emit_frame(self, *data, pace: bool = True) -> bool:
        """Waits until the GUI is ready (and for the playback speed limit), then emits the frame."""
        while not self._frame_slot.acquire(timeout=0.1):
            with self._lock:
                if not self.active:
                    return False
        if pace:
            wait = self._last_emit + self.min_frame_interval - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        self._last_emit = time.perf_counter()
        self.data_updated.emit(*data)
        return True

    def run(self):
        while True:
            with self._lock:
                if not self.active:
                    return
            idle = self._target is None and not self.playing
            try:
                command = self._commands.get(timeout=0.1) if idle else self._commands.get_nowait()
                self._apply(command)
                continue  # apply all queued commands first
            except queue.Empty:
                if idle:
                    continue
            if self._target is None:
                self._target = self.current + 1
            index, self._target = self._target, None
            img = self.source.get(index) if self.seekable else self.source.get(self.current + 1)
            if img is None:  # end of the video / image sequence
                self._set_playing(False)
                continue
            humans, objects, fps = self.process(index, img)
            if not self.emit_frame(index + 1, img, humans, objects, fps, pace=self.playing):
                return
            self.current = index
            self.position_changed.emit(index, max(self.source.frame_count, 0))

    def process(self, index: int, img: np.ndarray):
        """:return: (humans, object bbs, fps) of the frame"""
        raise NotImplementedError
