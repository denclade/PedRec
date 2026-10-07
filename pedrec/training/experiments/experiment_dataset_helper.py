from typing import List

from torch.utils.data import DataLoader, ConcatDataset

from pedrec.datasets.coco_dataset import CocoDataset
from pedrec.datasets.dataset_helper import worker_init_fn
from pedrec.datasets.pedrec_dataset import PedRecDataset
from pedrec.datasets.tud_dataset import TudDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.experiments.dataset_description import DatasetDescription
from pedrec.models.experiments.experiment_description import ExperimentDescription
from pedrec.models.experiments.validation_set_model import ValidationSet
from pedrec.training.experiments.experiment_train_helper import get_subsampled_dataset


def get_train_loader(experiment_description: ExperimentDescription, trans) -> DataLoader:
    train_sets = []
    if experiment_description.use_train_coco:
        coco_train = CocoDataset(experiment_description.experiment_paths.coco_dir, DatasetType.TRAIN,
                                 experiment_description.coco_train_dataset_cfg,
                                 experiment_description.net_cfg.model.input_size, trans)
        coco_full_length = len(coco_train)
        coco_train = get_subsampled_dataset(coco_train, experiment_description.coco_train_subsampling)
        train_sets.append(coco_train)
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
        experiment_description._train_sets.append(
            DatasetDescription(name=experiment_description.experiment_paths.h36m_train_filename,
                               subsampling=h36m_train.info.subsampling,
                               full_length=h36m_train.info.full_length,
                               used_length=h36m_train.info.used_length))

    if len(train_sets) == 0:
        raise ValueError("No training set! Check experiment config")

    train_set = ConcatDataset(train_sets)
    train_loader = DataLoader(train_set, batch_size=experiment_description.batch_size, shuffle=True, num_workers=experiment_description.num_workers, pin_memory=True,
                              persistent_workers=experiment_description.num_workers > 0,
                              worker_init_fn=worker_init_fn)
    return train_loader


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
    return validation_sets
