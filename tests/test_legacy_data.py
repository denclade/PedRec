"""The published dataframes were written with pandas 1.3 / numpy 1.21 (fixtures: tests/data/make_legacy_fixtures.py)."""
import numpy as np
import pandas as pd

from pedrec.configs.app_config import AppConfig
from pedrec.configs.dataset_configs import get_sim_dataset_cfg_default
from pedrec.datasets.pedrec_df_loader import get_annotations_from_pedrec_df
from pedrec.datasets.pedrec_temporal_dataset import PedRecTemporalDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.data_structures import ImageSize
from pedrec.training.experiments.ehpi3d_variants import VARIANTS, get_variant
from pedrec.utils.ehpi_helper import ehpi_transform
from pedrec.utils.pandas_helper import read_pedrec_df


def test_read_legacy_df_converts_numeric_categories(data_dir):
    df = read_pedrec_df(f"{data_dir}/legacy_pedrec_gt_df.pkl")
    assert df.shape == (40, 323)
    assert not isinstance(df["skeleton2d_nose_visible"].dtype, pd.CategoricalDtype)
    assert isinstance(df["img_dir"].dtype, pd.CategoricalDtype)  # string categories stay categorical
    assert df["actions"][0] == [1, 23]
    assert df["skeleton2d_nose_x"].dtype == np.float32


def test_annotations_from_legacy_df(data_dir):
    cfg = get_sim_dataset_cfg_default()
    index_mappings, annotations, info = get_annotations_from_pedrec_df(
        f"{data_dir}/legacy_pedrec_gt_df.pkl", cfg, ImageSize(192, 256), f"{data_dir}/legacy_pedrec_result_df.pkl")
    assert len(index_mappings) == 40
    assert annotations.skeleton2ds.shape == (40, 26, 5)
    assert annotations.skeleton3ds_results.shape == (40, 26, 6)
    assert info.provides_skeleton_3ds


def test_temporal_dataset_all_variants(data_dir):
    action_list = AppConfig().inference.action_list
    for name in ["gt", "pred_15fps", "gt_pred_64frames", "gt_pred_no_unit_skeleton", "gt_pred_zero_by_score"]:
        variant = get_variant(name)
        dataset = PedRecTemporalDataset(data_dir, "legacy_pedrec_gt_df.pkl", DatasetType.TRAIN, variant.get_pedrec_cfg(),
                                        action_list, ehpi_transform, pose_results_file="legacy_pedrec_result_df.pkl")
        ehpi, label = dataset[25]
        assert tuple(ehpi.shape) == (3, 32, variant.temporal_field[0])
        assert label.shape == (len(action_list),)
        assert label.sum() >= 1
    assert len(VARIANTS) == 18
