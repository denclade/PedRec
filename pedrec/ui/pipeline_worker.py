from typing import Optional

import numpy as np
from qtpy.QtWidgets import QApplication

from pedrec.inference.pipeline import PedRecPipeline
from pedrec.ui.pedrec_worker import PedRecWorker
from pedrec.utils.input_providers.input_provider_base import InputProviderBase


class PipelineWorker(PedRecWorker):
    """
    Qt worker thread which feeds the frames of an input provider through the PedRec pipeline and emits the results.
    """

    def __init__(self, parent: QApplication, input_provider: InputProviderBase, pipeline: PedRecPipeline,
                 max_fps: Optional[float] = None):
        super().__init__(parent, max_fps)
        self.input_provider = input_provider
        self.pipeline = pipeline

    def run_impl(self, frame_nr: int, img: np.ndarray):
        result = self.pipeline.process(frame_nr, img)
        self.emit_frame(frame_nr, img, result.humans, np.array(result.objects), result.fps)
