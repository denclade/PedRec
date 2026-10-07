from collections import OrderedDict
from typing import Optional

import numpy as np
from qtpy.QtWidgets import QApplication

from pedrec.inference.pipeline import PedRecPipeline
from pedrec.ui.pedrec_worker import PedRecWorker


class PipelineWorker(PedRecWorker):
    """
    Runs the PedRec pipeline for the player. The results of processed frames are cached, so stepping back, seeking
    into the processed part and replaying show the same (tracked) results instantly. The pipeline processes the
    frames in order; after a jump into an unprocessed part its temporal state (tracks, history) is reset.
    """

    def __init__(self, parent: QApplication, source, pipeline: PedRecPipeline, max_fps: Optional[float] = None,
                 cache_size: int = 5000):
        super().__init__(parent, source, max_fps)
        self.pipeline = pipeline
        self.cache: "OrderedDict[int, tuple]" = OrderedDict()
        self.cache_size = cache_size
        self.last_processed: Optional[int] = None

    def process(self, index: int, img: np.ndarray):
        if self.seekable and index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]
        if self.last_processed is not None and index != self.last_processed + 1:
            self.pipeline.reset()
        result = self.pipeline.process(index + 1, img)
        entry = (result.humans, np.array(result.objects), result.fps)
        self.last_processed = index
        if self.seekable:
            self.cache[index] = entry
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return entry
