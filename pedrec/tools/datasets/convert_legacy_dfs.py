"""
Rewrites PedRec dataset / result dataframes (``*.pkl``) with the currently installed pandas / numpy versions.

The published dataframes were pickled with pandas 1.3 / numpy 1.21. They can still be read (see
``pedrec.utils.pandas_helper.read_pedrec_df``), but numpy only keeps the ``numpy.core`` alias the old pickles refer to
for backwards compatibility. Converting them once makes the data independent of that alias. Values and columns are
unchanged; numeric categorical columns are stored with their numeric dtype. A ``.bak`` copy of the original file is
kept unless ``--no-backup`` is given.

    python pedrec/tools/datasets/convert_legacy_dfs.py data/datasets/ROMb/rt_rom_01b.pkl data/datasets/Conti01/*.pkl
"""
import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import os
import shutil

import pandas as pd

from pedrec.utils.pandas_helper import read_pedrec_df


def convert(path: str, backup: bool = True):
    df = read_pedrec_df(path)
    if backup:
        shutil.copy2(path, path + ".bak")
    tmp_path = path + ".tmp"
    df.to_pickle(tmp_path)
    check = pd.read_pickle(tmp_path)
    if check.shape != df.shape or list(check.columns) != list(df.columns):
        os.remove(tmp_path)
        raise RuntimeError(f"Verification of the converted file failed for {path}")
    os.replace(tmp_path, path)
    print(f"converted {path} ({df.shape[0]} rows, {df.shape[1]} columns)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="Dataframe pickles to convert in place.")
    parser.add_argument("--no-backup", action="store_true", help="Do not keep a .bak copy of the original file.")
    args = parser.parse_args(argv)
    for path in args.paths:
        convert(path, backup=not args.no_backup)


if __name__ == "__main__":
    main()
