import logging
import os
from typing import Dict, List

import torch

from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from pedrec.datasets.coco_dataset import CocoDataset
from pedrec.configs.dataset_configs import get_h36m_dataset_cfg_default
from pedrec.datasets.dataset_helper import worker_init_fn
from pedrec.datasets.extra_3d_datasets import EXTRA_3D_DATASETS
from pedrec.datasets.pedrec_dataset import PedRecDataset
from pedrec.datasets.tud_dataset import TudDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.experiments.dataset_description import DatasetDescription
from pedrec.models.experiments.experiment_description import ExperimentDescription
from pedrec.models.experiments.validation_set_model import ValidationSet
from pedrec.training.experiments.experiment_train_helper import get_subsampled_dataset

logger = logging.getLogger(__name__)


def get_train_loader(experiment_description: ExperimentDescription, trans) -> DataLoader:
    train_sets = []
    train_set_names = []
    if experiment_description.use_train_coco:
        coco_train = CocoDataset(experiment_description.experiment_paths.coco_dir, DatasetType.TRAIN,
                                 experiment_description.coco_train_dataset_cfg,
                                 experiment_description.net_cfg.model.input_size, trans)
        coco_full_length = len(coco_train)
        coco_train = get_subsampled_dataset(coco_train, experiment_description.coco_train_subsampling)
        train_sets.append(coco_train)
        train_set_names.append("coco")
        experiment_description._train_sets.append(
            DatasetDescription(name="COCO (TRAIN)",
                               subsampling=experiment_description.coco_train_subsampling,
                               full_length=coco_full_length,
                               used_length=len(coco_train)))
    if experiment_description.use_train_tud:
        tud_train = TudDataset(experiment_description.experiment_paths.tud_dir, DatasetType.TRAIN,
                               experiment_description.tud_train_dataset_cfg,
                               experiment_description.net_cfg.model.input_size, trans)
        tud_full_length = len(tud_train)
        tud_train = get_subsampled_dataset(tud_train, experiment_description.tud_train_subsampling)
        train_sets.append(tud_train)
        train_set_names.append("tud")
        experiment_description._train_sets.append(
            DatasetDescription(name="TUD (TRAIN)",
                               subsampling=experiment_description.tud_train_subsampling,
                               full_length=tud_full_length,
                               used_length=len(tud_train)))
    if experiment_description.use_train_sim:
        sim_train_a = PedRecDataset(experiment_description.experiment_paths.sim_train_dir,
                                  experiment_description.experiment_paths.sim_train_filename,
                                  DatasetType.TRAIN, experiment_description.sim_train_dataset_cfg,
                                  experiment_description.net_cfg.model.input_size, trans)
        train_sets.append(sim_train_a)
        train_set_names.append("sim")
        experiment_description._train_sets.append(
            DatasetDescription(name=experiment_description.experiment_paths.sim_train_filename,
                               subsampling=sim_train_a.info.subsampling,
                               full_length=sim_train_a.info.full_length,
                               used_length=sim_train_a.info.used_length))
        sim_train_b = PedRecDataset(experiment_description.experiment_paths.sim_val_dir,
                                  experiment_description.experiment_paths.sim_val_filename,
                                  DatasetType.TRAIN, experiment_description.sim_train_dataset_cfg,
                                  experiment_description.net_cfg.model.input_size, trans)
        train_sets.append(sim_train_b)
        train_set_names.append("sim")
        experiment_description._train_sets.append(
            DatasetDescription(name=experiment_description.experiment_paths.sim_val_filename,
                               subsampling=sim_train_b.info.subsampling,
                               full_length=sim_train_b.info.full_length,
                               used_length=sim_train_b.info.used_length))
    if experiment_description.use_train_h36m:
        h36m_train = PedRecDataset(experiment_description.experiment_paths.h36m_train_dir,
                                   experiment_description.experiment_paths.h36m_train_filename,
                                   DatasetType.TRAIN, experiment_description.h36m_train_dataset_cfg,
                                   experiment_description.net_cfg.model.input_size, trans)
        train_sets.append(h36m_train)
        train_set_names.append("h36m")
        experiment_description._train_sets.append(
            DatasetDescription(name=experiment_description.experiment_paths.h36m_train_filename,
                               subsampling=h36m_train.info.subsampling,
                               full_length=h36m_train.info.full_length,
                               used_length=h36m_train.info.used_length))

    if experiment_description.use_extra_3d:
        for extra in EXTRA_3D_DATASETS:
            path = extra.path(experiment_description.experiment_paths.datasets_dir, "train", "pedrec")
            if not extra.has_images or path is None:
                continue
            dataset = PedRecDataset(os.path.dirname(path), os.path.basename(path), DatasetType.TRAIN,
                                    get_extra_3d_dataset_cfg(), experiment_description.net_cfg.model.input_size, trans)
            train_sets.append(dataset)
            train_set_names.append(extra.name)
            experiment_description._train_sets.append(
                DatasetDescription(name=f"{extra.title} (TRAIN)", subsampling=dataset.info.subsampling,
                                   full_length=dataset.info.full_length, used_length=dataset.info.used_length))
            logger.info(f"Additional 3D dataset {extra.title}: {dataset.info.used_length} training samples")

    if len(train_sets) == 0:
        raise ValueError("No training set! Check experiment config")

    train_set = ConcatDataset(train_sets)
    sampler = None
    if experiment_description.dataset_sampling_weights:
        sampler = get_dataset_balancing_sampler(train_sets, train_set_names,
                                                experiment_description.dataset_sampling_weights)
    train_loader = DataLoader(train_set, batch_size=experiment_description.batch_size, shuffle=sampler is None,
                              sampler=sampler, num_workers=experiment_description.num_workers, pin_memory=True,
                              persistent_workers=experiment_description.num_workers > 0,
                              worker_init_fn=worker_init_fn, drop_last=True)
    return train_loader


def get_extra_3d_dataset_cfg(subsample: int = 1):
    """Additional 3D datasets: like Human3.6m (images img_<id>.jpg, no rotation augmentation of the 3D labels)."""
    cfg = get_h36m_dataset_cfg_default()
    cfg.subsample = subsample
    return cfg


def get_dataset_balancing_sampler(train_sets, names: List[str], weights: Dict[str, float]) -> WeightedRandomSampler:
    """
    Samples the datasets with the given relative probabilities, independent of their sizes (e.g. so that the large
    simulation datasets do not dominate COCO). One epoch has as many samples as all datasets together.
    """
    sample_weights = []
    for dataset, name in zip(train_sets, names):
        weight = weights.get(name, 1.0)
        sample_weights.append(torch.full((len(dataset),), weight / max(len(dataset), 1), dtype=torch.double))
    sample_weights = torch.cat(sample_weights)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def get_validation_sets(experiment_description: ExperimentDescription, trans) -> List[ValidationSet]:
    validation_sets: List[ValidationSet] = []
    if experiment_description.use_val_coco:
        coco_val = CocoDataset(experiment_description.experiment_paths.coco_dir, DatasetType.VALIDATE,
                               experiment_description.coco_val_dataset_cfg,
                               experiment_description.net_cfg.model.input_size,
                               trans)
        coco_val_full_length = len(coco_val)
        coco_val = get_subsampled_dataset(coco_val, experiment_description.coco_val_subsampling)
        coco_loader = DataLoader(coco_val, batch_size=experiment_description.batch_size_validate, shuffle=False,
                                 num_workers=experiment_description.num_workers, pin_memory=True,
                              persistent_workers=experiment_description.num_workers > 0, worker_init_fn=worker_init_fn)
        validation_sets.append(ValidationSet(name="COCO", loader=coco_loader,
                                             val_set_cfg=None,
                                             validate_2D=experiment_description.validate_2d_coco,
                                             validate_pose_conf=experiment_description.validate_joint_conf_coco,
                                             validate_orientation=experiment_description.validate_orientation_coco))
        experiment_description._val_sets.append(
            DatasetDescription(name="COCO (VAL)",
                               subsampling=experiment_description.coco_val_subsampling,
                               full_length=coco_val_full_length,
                               used_length=len(coco_val)))
    if experiment_description.use_val_tud:
        tud_val = TudDataset(experiment_description.experiment_paths.tud_dir, DatasetType.VALIDATE,
                               experiment_description.tud_val_dataset_cfg,
                               experiment_description.net_cfg.model.input_size,
                               trans)
        tud_val_full_length = len(tud_val)
        tud_val = get_subsampled_dataset(tud_val, experiment_description.tud_val_subsampling)
        tud_loader = DataLoader(tud_val, batch_size=experiment_description.batch_size_validate, shuffle=False,
                                 num_workers=experiment_description.num_workers, pin_memory=True,
                              persistent_workers=experiment_description.num_workers > 0, worker_init_fn=worker_init_fn)
        validation_sets.append(ValidationSet(name="TUD", loader=tud_loader,
                                             val_set_cfg=None,
                                             validate_2D=False,
                                             validate_pose_conf=False,
                                             validate_orientation=experiment_description.validate_orientation_tud))
        experiment_description._val_sets.append(
            DatasetDescription(name="TUD (VAL)",
                               subsampling=experiment_description.tud_val_subsampling,
                               full_length=tud_val_full_length,
                               used_length=len(tud_val)))
    if experiment_description.use_val_sim:
        sim_val = PedRecDataset(experiment_description.experiment_paths.sim_val_dir,
                                experiment_description.experiment_paths.sim_val_filename, DatasetType.VALIDATE,
                                experiment_description.sim_val_dataset_cfg,
                                experiment_description.net_cfg.model.input_size,
                                trans)
        sim_loader = DataLoader(sim_val, batch_size=experiment_description.batch_size_validate, shuffle=False,
                                num_workers=experiment_description.num_workers, pin_memory=True,
                              persistent_workers=experiment_description.num_workers > 0, worker_init_fn=worker_init_fn)
        validation_sets.append(ValidationSet(name="SIM", loader=sim_loader,
                                             val_set_cfg=experiment_description.sim_val_dataset_cfg,
                                             validate_2D=experiment_description.validate_2d_sim,
                                             validate_3D=experiment_description.validate_3d_sim,
                                             validate_pose_conf=experiment_description.validate_joint_conf_sim,
                                             validate_orientation=experiment_description.validate_orientation_sim,
                                             validate_env_position=experiment_description.validate_env_position_sim))
        experiment_description._val_sets.append(
            DatasetDescription(name=experiment_description.experiment_paths.sim_val_filename,
                               subsampling=sim_val.info.subsampling,
                               full_length=sim_val.info.full_length,
                               used_length=sim_val.info.used_length))
    if experiment_description.use_val_h36m:
        h36m_val = PedRecDataset(experiment_description.experiment_paths.h36m_val_dir,
                                 experiment_description.experiment_paths.h36m_val_filename,
                                 DatasetType.VALIDATE, experiment_description.h36m_val_dataset_cfg,
                                 experiment_description.net_cfg.model.input_size, trans)
        h36m_loader = DataLoader(h36m_val, batch_size=experiment_description.batch_size_validate, shuffle=False,
                                 num_workers=experiment_description.num_workers, pin_memory=True,
                              persistent_workers=experiment_description.num_workers > 0, worker_init_fn=worker_init_fn)
        validation_sets.append(ValidationSet(name="H36M", loader=h36m_loader,
                                             val_set_cfg=experiment_description.h36m_val_dataset_cfg,
                                             validate_2D=experiment_description.validate_2d_h36m,
                                             validate_3D=experiment_description.validate_3d_h36m,
                                             validate_pose_conf=experiment_description.validate_joint_conf_h36m,
                                             validate_env_position=experiment_description.validate_env_position_h36m))
        experiment_description._val_sets.append(
            DatasetDescription(name=experiment_description.experiment_paths.h36m_val_filename,
                               subsampling=h36m_val.info.subsampling,
                               full_length=h36m_val.info.full_length,
                               used_length=h36m_val.info.used_length))
    if experiment_description.use_extra_3d:
        for extra in EXTRA_3D_DATASETS:
            path = extra.path(experiment_description.experiment_paths.datasets_dir, "val", "pedrec")
            if not extra.has_images or path is None:
                continue
            val_cfg = get_extra_3d_dataset_cfg(extra.val_subsample)
            dataset = PedRecDataset(os.path.dirname(path), os.path.basename(path), DatasetType.VALIDATE, val_cfg,
                                    experiment_description.net_cfg.model.input_size, trans)
            loader = DataLoader(dataset, batch_size=experiment_description.batch_size_validate, shuffle=False,
                                num_workers=experiment_description.num_workers, pin_memory=True,
                                persistent_workers=experiment_description.num_workers > 0,
                                worker_init_fn=worker_init_fn)
            validation_sets.append(ValidationSet(name=extra.title, loader=loader, val_set_cfg=val_cfg,
                                                 validate_2D=True, validate_3D=experiment_description.validate_3d_h36m,
                                                 validate_pose_conf=False, validate_env_position=False))
            experiment_description._val_sets.append(
                DatasetDescription(name=f"{extra.title} (VAL)", subsampling=dataset.info.subsampling,
                                   full_length=dataset.info.full_length, used_length=dataset.info.used_length))
    return validation_sets
