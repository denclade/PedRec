"""
Additional 3D pose datasets converted to PedRec dataframes (``pedrec/tools/datasets/convert_*.py``, ``mise run
data:convert:<name>``). They are used automatically by the PedRecNet training (images) and the lifter training
(sequences) as soon as the converted files exist below the datasets directory.
"""
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Extra3DDataset:
    name: str  # short name (dataset balancing: --dataset-weights mpi_inf_3dhp=1,...)
    title: str
    directory: str  # below the datasets directory
    prefix: str  # <prefix>_<split>_pedrec.pkl (images) / <prefix>_<split>_seq.pkl (sequences)
    has_images: bool = True
    val_subsample: int = 4

    def path(self, datasets_dir: str, split: str, kind: str) -> Optional[str]:
        """:param kind: "pedrec" (frames with images) or "seq" (sequences for the lifter)"""
        path = os.path.join(datasets_dir, self.directory, f"{self.prefix}_{split}_{kind}.pkl")
        return path if os.path.isfile(path) else None


EXTRA_3D_DATASETS: List[Extra3DDataset] = [
    Extra3DDataset("mpi_inf_3dhp", "MPI-INF-3DHP", "MPI-INF-3DHP", "mpi_inf_3dhp"),
    Extra3DDataset("fit3d", "Fit3D", "Fit3D", "fit3d"),
    Extra3DDataset("aistpp", "AIST++", "AIST++", "aistpp"),
    Extra3DDataset("amass", "AMASS", "AMASS", "amass", has_images=False),
]
