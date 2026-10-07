import os
from dataclasses import dataclass


@dataclass()
class ExperimentPaths(object):
    """
    All dataset / checkpoint locations used by the training and evaluation scripts.
    Use ``pedrec.training.experiments.experiment_path_helper.get_experiment_paths`` to create an instance
    based on the data root (``PEDREC_DATA_DIR``).
    """
    pose_resnet_weights_path: str
    pose_2d_coco_only_weights_path: str
    pedrec_2d_path: str
    pedrec_2d_h36m_path: str
    pedrec_2d_sim_path: str
    pedrec_2d_c_path: str
    pedrec_2d_h36m_sim_path: str
    pedrec_2d3d_h36m_path: str
    pedrec_2d3d_sim_path: str
    pedrec_2d3d_h36m_sim_path: str
    pedrec_2d3d_c_h36m_path: str
    pedrec_2d3d_c_sim_path: str
    pedrec_2d3d_c_h36m_sim_path: str
    pedrec_2d3d_c_o_h36m_mebow_path: str
    pedrec_2d3d_c_o_sim_path: str
    pedrec_2d3d_c_o_h36m_sim_path: str
    pedrec_2d3d_c_o_h36m_sim_mebow_path: str
    pedrec_full_path: str
    output_dir: str
    coco_dir: str
    tud_dir: str
    sim_train_dir: str
    sim_val_dir: str
    h36m_train_dir: str
    h36m_val_dir: str
    sim_c01_dir: str
    sim_c01_filename: str
    sim_c01_results_filename: str
    sim_c01_val_dir: str
    sim_c01_val_filename: str
    sim_c01_val_results_filename: str
    pretrained_model_path: str = None
    sim_train_filename: str = "rt_rom_01b.pkl"
    sim_val_filename: str = "rt_validate_3d.pkl"
    h36m_val_filename: str = "h36m_val_pedrec.pkl"
    h36m_train_filename: str = "h36m_train_pedrec.pkl"
    ehpi3d_output_dir: str = "data/models/ehpi3d"
    ehpi_videos_dir: str = "data/videos/ehpi_videos"
    ehpi_videos_results_filename: str = "pedrec_p2d3d_c_o_h36m_sim_mebow_0_results.pkl"
    checkpoint_prefix: str = "experiment_pedrec"  # e.g. experiment_pedrec_v2 for other network architectures

    def get_stage_file_base(self, stage_name: str) -> str:
        return os.path.join(self.output_dir, f"{self.checkpoint_prefix}_{stage_name}")

    def get_stage_checkpoint_path(self, stage_name: str, cycle_num: int = 0, round_suffix: str = None) -> str:
        """
        Path of the (MTL wrapper) checkpoint written by ``train_pedrec.py`` for a training stage.
        """
        filename = f"{self.get_stage_file_base(stage_name)}_{cycle_num}"
        if round_suffix is not None:
            filename += f"_{round_suffix}"
        return f"{filename}.pth"

    def get_stage_protocol_path(self, stage_name: str) -> str:
        return f"{self.get_stage_file_base(stage_name)}_protocol.md"
