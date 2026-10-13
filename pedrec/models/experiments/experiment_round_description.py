from dataclasses import dataclass
from typing import Dict, List

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from pedrec.models.experiments.epoch_validation_results import EpochValidationResults


@dataclass()
class ExperimentRoundDescription(object):
    num_epochs: int
    optimizer: Optimizer
    scheduler: _LRScheduler
    optimizer_parameters: Dict[str, any]
    max_lrs: List[float]
    frozen_layers: List[nn.Module]
    validation_results: List[EpochValidationResults] = None
    name: str = ""  # for the terminal output, e.g. "Round 1/2 (backbone frozen)"
