import os

from pedrec.configs.default_paths import get_data_root
from pedrec.models.experiments.experiment_paths import ExperimentPaths


def get_experiment_paths(data_root: str = None) -> ExperimentPaths:
    """
    Builds the default dataset / checkpoint layout below the data root (``PEDREC_DATA_DIR``, default ``data``).
    Override individual attributes on the returned object if your layout differs.
    """
    root = get_data_root(data_root)
    datasets = os.path.join(root, "datasets")
    models = os.path.join(root, "models")
    checkpoints = os.path.join(models, "pedrec", "single_results")

    def checkpoint(name: str) -> str:
        return os.path.join(checkpoints, f"experiment_pedrec_{name}_0.pth")

    return ExperimentPaths(
        pose_resnet_weights_path=os.path.join(models, "human_pose_baseline", "pose_resnet_50_256x192.pth.tar"),
        pose_2d_coco_only_weights_path=checkpoint("p2d_coco_only"),
        output_dir=checkpoints,
        coco_dir=os.path.join(datasets, "COCO"),
        tud_dir=os.path.join(datasets, "cvpr10_multiview_pedestrians"),
        sim_train_dir=os.path.join(datasets, "ROMb"),
        sim_val_dir=os.path.join(datasets, "RT3DValidate"),
        h36m_train_dir=os.path.join(datasets, "Human3.6m", "train"),
        h36m_val_dir=os.path.join(datasets, "Human3.6m", "val"),
        sim_c01_dir=os.path.join(datasets, "Conti01"),
        sim_c01_filename="rt_conti_01_train_FIN.pkl",
        sim_c01_results_filename="C01F_train_pred_df_experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_allframes.pkl",
        sim_c01_val_dir=os.path.join(datasets, "Conti01"),
        sim_c01_val_filename="rt_conti_01_val_FIN.pkl",
        sim_c01_val_results_filename="C01F_pred_df_experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_allframes.pkl",

        pedrec_2d_path=checkpoint("p2d_coco_only"),
        pedrec_2d_c_path=checkpoint("p2d_c"),
        pedrec_2d_h36m_path=checkpoint("p2d_h36m"),
        pedrec_2d_sim_path=checkpoint("p2d_sim"),
        pedrec_2d_h36m_sim_path=checkpoint("p2d_h36m_sim"),
        pedrec_2d3d_h36m_path=checkpoint("p2d3d_h36m"),
        pedrec_2d3d_sim_path=checkpoint("p2d3d_sim"),
        pedrec_2d3d_h36m_sim_path=checkpoint("p2d3d_h36m_sim"),
        pedrec_2d3d_c_h36m_path=checkpoint("p2d3d_c_h36m"),
        pedrec_2d3d_c_sim_path=checkpoint("p2d3d_c_sim"),
        pedrec_2d3d_c_h36m_sim_path=checkpoint("p2d3d_c_h36m_sim"),
        pedrec_2d3d_c_o_h36m_mebow_path=checkpoint("p2d3d_c_o_h36m_mebow"),
        pedrec_2d3d_c_o_sim_path=checkpoint("p2d3d_c_o_sim"),
        pedrec_2d3d_c_o_h36m_sim_path=checkpoint("p2d3d_c_o_h36m_sim"),
        pedrec_2d3d_c_o_h36m_sim_mebow_path=checkpoint("p2d3d_c_o_h36m_sim_mebow"),
        pedrec_full_path=checkpoint("p2d3d_c_o_h36m_sim"),
        ehpi3d_output_dir=os.path.join(models, "ehpi3d"),
        ehpi_videos_dir=os.path.join(root, "videos", "ehpi_videos"),
    )


def get_experiment_paths_home() -> ExperimentPaths:
    """
    Deprecated alias kept for the older evaluation / tool scripts. Use ``get_experiment_paths``.
    """
    return get_experiment_paths()
