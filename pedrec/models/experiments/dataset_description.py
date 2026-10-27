from dataclasses import dataclass
from typing import Optional


@dataclass()
class DatasetDescription(object):
    name: str
    subsampling: int
    full_length: int
    used_length: int
    sampling_share: Optional[float] = None  # share of the training samples (dataset balancing), None = by size
