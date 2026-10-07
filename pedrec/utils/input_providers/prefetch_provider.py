import queue
import threading
from typing import Iterator, Optional

import numpy as np

from pedrec.utils.input_providers.input_provider_base import InputProviderBase

_END = object()


class PrefetchProvider(InputProviderBase):
    """
    Wraps another input provider and reads / decodes its frames in a background thread, so that video decoding and
    resizing overlap with the network inference.
    """

    def __init__(self, provider: InputProviderBase, queue_size: int = 4):
        self.provider = provider
        self.queue: "queue.Queue" = queue.Queue(maxsize=max(1, queue_size))
        self.stopped = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[BaseException] = None

    def _reader(self):
        try:
            for frame in self.provider.get_data():
                if self.stopped.is_set():
                    break
                while not self.stopped.is_set():
                    try:
                        self.queue.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:  # forwarded to the consumer
            self.error = e
        finally:
            while True:
                try:
                    self.queue.put(_END, timeout=0.1)
                    break
                except queue.Full:
                    if self.stopped.is_set():
                        break

    def get_data(self) -> Iterator[np.ndarray]:
        self.thread = threading.Thread(target=self._reader, name="pedrec-prefetch", daemon=True)
        self.thread.start()
        while True:
            frame = self.queue.get()
            if frame is _END:
                break
            yield frame
        if self.error is not None:
            raise self.error

    def stop(self):
        self.stopped.set()
        # let the reader finish its current frame before the underlying capture is released
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=5)
        if hasattr(self.provider, "stop"):
            self.provider.stop()
