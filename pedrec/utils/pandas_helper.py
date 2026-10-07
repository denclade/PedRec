import math
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from pedrec.configs.dataset_configs import PedRecDatasetConfig
from pedrec.models.constants.sample_method import SAMPLE_METHOD


def normalize_df_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts categorical columns with numeric categories (e.g. ``*_visible``, ``*_supported``, ``scene_id``, ...)
    into their plain numeric dtype.

    The published PedRec dataframes were written with pandas 1.3 and store many numeric flags as ``category``.
    Current pandas versions do not support arithmetic reductions (``sum``, ``mean``, ...) on categorical data, so these
    columns are converted once after loading. The values are unchanged; string categories (``img_dir``,
    ``subject_id``, ...) stay categorical.
    """
    conversions = {}
    for col, dtype in df.dtypes.items():
        if isinstance(dtype, pd.CategoricalDtype) and is_numeric_dtype(dtype.categories.dtype):
            conversions[col] = dtype.categories.dtype
    if conversions:
        df = df.astype(conversions)
    return df.copy()  # consolidate the blocks (the original frames are highly fragmented)


def read_pedrec_df(df_path: str) -> pd.DataFrame:
    """
    Reads a PedRec dataset / result dataframe (pickle) independent of the pandas version it was written with.

    Pickles written with numpy 1.x reference ``numpy.core`` which numpy 2 only provides as a deprecated alias. Use
    ``pedrec/tools/datasets/convert_legacy_dfs.py`` to rewrite the files once with the current versions.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"numpy\.core.*", category=DeprecationWarning)
        df = pd.read_pickle(df_path)
    return normalize_df_dtypes(df)


def get_subsampled_df(df_path: str, cfg: PedRecDatasetConfig):
    df = read_pedrec_df(df_path)

    df_full_length = len(df)
    if cfg.subsample != 1:
        if cfg.subsampling_strategy == SAMPLE_METHOD.SYSTEMATIC:
            df = df.loc[range(0, df_full_length, cfg.subsample)]
        elif cfg.subsampling_strategy == SAMPLE_METHOD.RANDOM:
            df = df.loc[np.random.choice(df.index, math.floor(len(df) / cfg.subsample), replace=False)]
        else:
            raise NotImplementedError(f"Sampling strategy {cfg.subsampling_strategy.name} is not implemented.")
    return df
