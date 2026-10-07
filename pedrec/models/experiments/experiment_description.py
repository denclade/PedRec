from dataclasses import dataclass, field
from typing import Dict, List, Optional

from torch import nn


from pedrec.configs.dataset_configs import CocoDatasetConfig, PedRecDatasetConfig, get_coco_dataset_cfg_default, \
    get_sim_dataset_cfg_default, get_h36m_dataset_cfg_default, get_h36m_val_dataset_cfg_default, \
    get_sim_val_dataset_cfg_default, get_tud_dataset_cfg_default, TudDatasetConfig
from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.models.experiments.dataset_description import DatasetDescription
from pedrec.models.experiments.experiment_round_description import ExperimentRoundDescription
from pedrec.training.experiments.experiment_path_helper import ExperimentPaths

@dataclass()
class ExperimentDescription(object):
    net_name: str
    experiment_name: str
    experiment_paths: ExperimentPaths
    net_cfg: PedRecNetConfig
    initialization_notes: str
    coco_train_dataset_cfg: CocoDatasetConfig = field(default_factory=get_coco_dataset_cfg_default)
    tud_train_dataset_cfg: TudDatasetConfig = field(default_factory=get_tud_dataset_cfg_default)
    sim_train_dataset_cfg: PedRecDatasetConfig = field(default_factory=get_sim_dataset_cfg_default)
    h36m_train_dataset_cfg: PedRecDatasetConfig = field(default_factory=get_h36m_dataset_cfg_default)
    coco_val_dataset_cfg: CocoDatasetConfig = field(default_factory=get_coco_dataset_cfg_default)
    tud_val_dataset_cfg: TudDatasetConfig = field(default_factory=get_tud_dataset_cfg_default)
    sim_val_dataset_cfg: PedRecDatasetConfig = field(default_factory=get_sim_val_dataset_cfg_default)
    h36m_val_dataset_cfg: PedRecDatasetConfig = field(default_factory=get_h36m_val_dataset_cfg_default)
    seed: int = 42
    batch_size: int = 48
    batch_size_validate: int = 48
    num_workers: int = 12
    dataset_sampling_weights: Optional[Dict[str, float]] = None  # e.g. {'coco': 1, 'h36m': 1, 'sim': 2}

    use_train_coco: bool = True
    use_train_sim: bool = True
    use_train_h36m: bool = True
    use_train_tud: bool = False
    coco_train_subsampling: int = 1
    h36m_train_subsampling: int = 10
    sim_train_subsampling: int = 1
    tud_train_subsampling: int = 1

    use_val_coco: bool = True
    use_val_sim: bool = True
    use_val_h36m: bool = True
    use_val_tud: bool = False
    coco_val_subsampling: int = 1
    h36m_val_subsampling: int = 64
    sim_val_subsampling: int = 5
    tud_val_subsampling: int = 1

    validate_2d_coco: bool = True
    validate_2d_sim: bool = True
    validate_2d_h36m: bool = True

    validate_3d_sim: bool = True
    validate_3d_h36m: bool = True

    validate_joint_conf_coco: bool = False
    validate_joint_conf_sim: bool = False
    validate_joint_conf_h36m: bool = False

    validate_env_position_sim: bool = False
    validate_env_position_h36m: bool = False
    validate_orientation_sim: bool = False
    validate_orientation_coco: bool = False
    validate_orientation_tud: bool = False

    _train_sets: List[DatasetDescription] = field(default_factory=list)
    _val_sets: List[DatasetDescription] = field(default_factory=list)

    suggested_lr: float = 0.0
    net_layers: List[nn.Module] = field(default_factory=list)
    net_layer_names: List[str] = field(default_factory=list)
    experiment_rounds: List[ExperimentRoundDescription] = field(default_factory=list)


    def __str__(self):
        return f""