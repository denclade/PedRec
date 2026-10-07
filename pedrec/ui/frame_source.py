"""
Frame sources of the Qt player: random access to the frames of a video file, an image directory or a single image
(seeking, frame by frame), with read-ahead in a background thread so that decoding overlaps with the processing.
Webcams are live sources without seeking.
"""
import threading
from collections import OrderedDict
from typing import List, Optional

import cv2
import numpy as np

from pedrec.models.data_structures import ImageSize
from pedrec.utils.file_helper import get_img_paths_from_folder


def _prepare(frame_bgr: np.ndarray, img_size: ImageSize, mirror: bool) -> np.ndarray:
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if frame.shape[1] != img_size.width or frame.shape[0] != img_size.height:
        frame = cv2.resize(frame, (img_size.width, img_size.height))
    if mirror:
        frame = cv2.flip(frame, 1)
    return frame


class _VideoReader:
    def __init__(self, path: str, img_size: ImageSize, mirror: bool):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video {path}")
        self.img_size, self.mirror = img_size, mirror
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.position = 0

    def seek(self, index: int):
        if index != self.position:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            self.position = index

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        self.position += 1
        return _prepare(frame, self.img_size, self.mirror)

    def close(self):
        self.cap.release()


class _ImageListReader:
    def __init__(self, paths: List[str], img_size: ImageSize, mirror: bool):
        self.paths, self.img_size, self.mirror = paths, img_size, mirror
        self.frame_count = len(paths)
        self.position = 0

    def seek(self, index: int):
        self.position = index

    def read(self) -> Optional[np.ndarray]:
        if self.position >= len(self.paths):
            return None
        frame = cv2.imread(self.paths[self.position])
        self.position += 1
        return None if frame is None else _prepare(frame, self.img_size, self.mirror)

    def close(self):
        pass


class FrameSource:
    """
    Seekable frame source. ``get(index)`` returns the frame (RGB, ``img_size``) or None after the last frame. Frames
    after the requested one are read ahead in a background thread, a few previous ones are kept for stepping back.
    """
    seekable = True

    def __init__(self, reader, fps: float, read_ahead: int = 8, keep_previous: int = 12):
        self.reader = reader
        self.fps = fps
        self.frame_count = reader.frame_count  # may be an estimate for videos, the end is detected while reading
        self.read_ahead = read_ahead
        self.keep_previous = keep_previous
        self._buffer: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._next = 0  # index the reader thread reads next
        self._end: Optional[int] = None  # index of the first frame that does not exist
        self._seek_to: Optional[int] = None
        self._wanted = 0  # last requested index (read-ahead window starts here)
        self._generation = 0
        self._active = True
        self._cond = threading.Condition()
        self._thread = threading.Thread(target=self._run, name="pedrec-frame-source", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            with self._cond:
                while self._active and self._seek_to is None and (
                        self._end is not None or self._next >= self._wanted + self.read_ahead):
                    self._cond.wait()
                if not self._active:
                    break
                if self._seek_to is not None:
                    self.reader.seek(self._seek_to)
                    self._next, self._end, self._seek_to = self._seek_to, None, None
                    self._generation += 1
                index, generation = self._next, self._generation
            frame = self.reader.read()
            with self._cond:
                if generation == self._generation:
                    if frame is None:
                        self._end = index
                        if self.frame_count <= 0 or self.frame_count > index:
                            self.frame_count = index
                    else:
                        self._buffer[index] = frame
                        self._next = index + 1
                        self._trim()
                self._cond.notify_all()
        self.reader.close()

    def _trim(self):
        # keep keep_previous frames before the requested one and the read-ahead window after it
        for index in [i for i in self._buffer if i < self._wanted - self.keep_previous or
                      i >= self._wanted + self.read_ahead + 1]:
            del self._buffer[index]

    def get(self, index: int) -> Optional[np.ndarray]:
        if index < 0:
            return None
        with self._cond:
            self._wanted = index
            self._trim()
            in_window = index in self._buffer or self._next <= index < self._next + self.read_ahead
            if not in_window and (self._end is None or index < self._end):
                self._seek_to = index
                self._buffer.clear()
            self._cond.notify_all()
            while index not in self._buffer:
                if self._end is not None and index >= self._end and self._seek_to is None:
                    return None
                if not self._active:
                    return None
                self._cond.wait(timeout=1.0)
            return self._buffer[index]

    def stop(self):
        with self._cond:
            self._active = False
            self._cond.notify_all()
        self._thread.join(timeout=5)


class LiveSource:
    """Webcam: no seeking, ``get`` returns the next captured frame."""
    seekable = False

    def __init__(self, camera: int, img_size: ImageSize, mirror: bool, fps: float = 30.0):
        self.cap = cv2.VideoCapture(camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_size.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_size.height)
        self.img_size, self.mirror, self.fps = img_size, mirror, fps
        self.frame_count = 0

    def get(self, index: int) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return _prepare(frame, self.img_size, self.mirror) if ok and frame is not None else None

    def stop(self):
        self.cap.release()


def open_frame_source(img_size: ImageSize, video: str = None, images: str = None, image: str = None,
                      webcam: int = None, mirror: bool = False, fps: Optional[float] = None):
    if webcam is not None:
        return LiveSource(webcam, img_size, mirror, fps or 30.0)
    if video is not None:
        return FrameSource(_VideoReader(video, img_size, mirror), fps or 30.0)
    paths = sorted(get_img_paths_from_folder(images)) if images is not None else [image]
    return FrameSource(_ImageListReader(paths, img_size, mirror), fps or 30.0)
