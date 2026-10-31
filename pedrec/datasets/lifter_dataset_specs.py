"""Shared dataset/result locations for lifter data generation and training."""
import hashlib
import os
from dataclasses import dataclass

from pedrec.datasets.extra_3d_datasets import EXTRA_3D_DATASETS
from pedrec.networks.net_pedrec.pedrec_convention_adapter import get_convention_index


@dataclass(frozen=True)
class LifterDatasetSpec:
    name: str
    title: str
    split: str
    dataframe: str
    results: str
    convention: int = -1
    image_pattern: str = "img_{id}.{type}"
    has_images: bool = True
    dense_results: bool = False


def prediction_path(dataframe: str) -> str:
    return os.path.splitext(dataframe)[0] + "_predictions.pkl"


def dataset_specs(paths):
    """Existing sequence dataframes; results need not have been generated yet.

    SIM-C01 keeps its legacy dense filenames for the EHPI tooling. Other sets
    store only predicted rows with explicit original_index, avoiding huge zero
    dataframes for sequence frames whose images were never extracted.
    """
    specs = []
    for name, title, split, directory, filename, convention, pattern, results in [
        ("h36m", "H36M", "train", paths.h36m_train_dir, paths.h36m_train_filename, "h36m", "img_{id}.{type}", None),
        ("h36m", "H36M", "val", paths.h36m_val_dir, paths.h36m_val_filename, "h36m", "img_{id}.{type}", None),
        ("sim_rom", "SIM-ROM", "train", paths.sim_train_dir, paths.sim_train_filename, "sim", "view_{cam_name}-frame_{id}.{type}", None),
        ("sim_circle", "SIM-Circle", "val", paths.sim_val_dir, paths.sim_val_filename, "sim", "view_{cam_name}-frame_{id}.{type}", None),
        ("sim_c01", "SIM-C01", "train", paths.sim_c01_dir, paths.sim_c01_filename, "sim", "view_{cam_name}-frame_{id}.{type}", paths.sim_c01_results_filename),
        ("sim_c01", "SIM-C01", "val", paths.sim_c01_val_dir, paths.sim_c01_val_filename, "sim", "view_{cam_name}-frame_{id}.{type}", paths.sim_c01_val_results_filename),
    ]:
        dataframe = os.path.join(directory, filename)
        if os.path.isfile(dataframe):
            specs.append(LifterDatasetSpec(name, title, split, dataframe,
                                          os.path.join(directory, results) if results else prediction_path(dataframe),
                                          get_convention_index(convention), pattern, dense_results=results is not None))
    for extra in EXTRA_3D_DATASETS:
        for split in ("train", "val"):
            dataframe = extra.path(paths.datasets_dir, split, "seq")
            if dataframe:
                specs.append(LifterDatasetSpec(extra.name, extra.title, split, dataframe, prediction_path(dataframe),
                                              get_convention_index(extra.name), has_images=extra.has_images))
    return specs


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataframe_signature(path: str) -> dict:
    stat = os.stat(path)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
