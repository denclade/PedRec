"""Dataset check: downloaded / converted / missing detection, image samples at the training rows, next steps."""
import os

import numpy as np
import pandas as pd

from pedrec.tools.datasets import check_datasets as cd
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths


def _df(path, img_dirs, ids):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"img_dir": pd.Categorical(img_dirs), "img_id": np.asarray(ids, np.uint32),
                  "img_type": pd.Categorical(["jpg"] * len(ids))}).to_pickle(path)


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").close()


def _items(group):
    return {item.name: item for item in group.items}


def test_empty_root(tmp_path, capsys):
    assert cd.run(str(tmp_path), color=False) == 1
    out = capsys.readouterr().out
    assert "train:pedrec   not ready" in out and "mise run download:datasets:coco\n" in out


def test_h36m_images_at_training_rows(tmp_path):
    paths = get_experiment_paths(str(tmp_path))
    n = 100
    _df(os.path.join(paths.h36m_train_dir, paths.h36m_train_filename), ["S1/Images/Walk 1.54138969"] * n,
        range(1, n + 1))
    sampler = cd.ImageSampler(str(tmp_path / "cache.json"), samples=200)
    group = cd.check_pedrec_training(paths, paths.datasets_dir, sampler)
    assert _items(group)["Human3.6m train images"].status == cd.MISSING
    assert "info h36m" in _items(group)["Human3.6m train videos"].fix  # no videos yet
    # only every 10th row is read by the training -> only these images are needed
    for i in range(1, n + 1, 10):
        _touch(os.path.join(paths.h36m_train_dir, "S1", "Images", "Walk 1.54138969", f"img_{i:05d}.jpg"))
    _touch(os.path.join(paths.h36m_train_dir, "S1", "Videos", "Walk 1.54138969.mp4"))
    group = cd.check_pedrec_training(paths, paths.datasets_dir, sampler)
    assert _items(group)["Human3.6m train images"].status == cd.OK
    assert "10/10" in _items(group)["Human3.6m train images"].detail
    os.remove(os.path.join(paths.h36m_train_dir, "S1", "Images", "Walk 1.54138969", "img_00011.jpg"))
    group = cd.check_pedrec_training(paths, paths.datasets_dir, sampler)
    images = _items(group)["Human3.6m train images"]
    assert images.status == cd.PARTIAL and images.fix == "mise run data:h36m:images"
    sampler.save()
    assert os.path.isfile(tmp_path / "cache.json")


def test_ready_for_pedrec_training(tmp_path, monkeypatch, capsys):
    paths = get_experiment_paths(str(tmp_path))
    monkeypatch.setattr(cd, "COCO_IMAGES", {"train": 2, "val": 1})
    monkeypatch.setattr(cd, "check_models", lambda data_root: cd.Group("models"))
    for name in ("person_keypoints_train2017.json", "person_keypoints_val2017.json", "train_hoe.json",
                 "val_hoe.json"):
        _touch(os.path.join(paths.coco_dir, "annotations", name))
    for split, count in (("train", 2), ("val", 1)):
        for i in range(count):
            _touch(os.path.join(paths.coco_dir, f"{split}2017", f"{i:012d}.jpg"))
    for root, filename in ((paths.h36m_train_dir, paths.h36m_train_filename),
                           (paths.h36m_val_dir, paths.h36m_val_filename)):
        _df(os.path.join(root, filename), ["S9/Images/a"], [1])
        _touch(os.path.join(root, "S9", "Images", "a", "img_00001.jpg"))
    for root, filename in ((paths.sim_train_dir, paths.sim_train_filename), (paths.sim_val_dir, paths.sim_val_filename)):
        _df(os.path.join(root, filename), ["scene/cam1"], [3])
        _touch(os.path.join(root, "scene", "cam1", "view_cam1-frame_00003.jpg"))
    # AIST++ downloaded but not converted, Fit3D converted
    _touch(os.path.join(paths.datasets_dir, "AIST++", "annotations", "keypoints3d", "gBR_sBM_cAll_d04.pkl"))
    _df(os.path.join(paths.datasets_dir, "Fit3D", "fit3d_train_pedrec.pkl"), ["images/s03_squat_cam"], [1])
    _touch(os.path.join(paths.datasets_dir, "Fit3D", "images", "s03_squat_cam", "img_00001.jpg"))
    assert cd.run(str(tmp_path), color=False) == 0
    out = capsys.readouterr().out
    assert "train:pedrec   ready" in out and "train:all      not ready" in out
    assert "-> mise run data:convert:aistpp" in out  # downloaded, not converted
    assert "Fit3D images" in out and "AMASS with at least PosePrior" in out


def test_merge_steps():
    assert cd.merge_steps(["a", "mise run download:datasets:pedrec --parts h36m", "a",
                           "mise run download:datasets:pedrec --parts rom"]) == \
        ["a", "mise run download:datasets:pedrec --parts h36m rom"]
    assert cd.merge_steps([f"mise run download:datasets:coco --parts {p}" for p in ("annotations", "train", "val")]) \
        == ["mise run download:datasets:coco"]


def test_amass_range_of_motion_subset(tmp_path):
    datasets = tmp_path / "datasets"
    _df(str(datasets / "AMASS" / "amass_train_seq.pkl"), ["images/x"], [1])
    _touch(str(datasets / "AMASS" / "CMU" / "01" / "01_01_poses.npz"))
    sampler = cd.ImageSampler(os.devnull, samples=10)
    group, converted = cd.check_extra_3d(str(datasets), str(tmp_path), sampler)
    assert converted["amass"] and "AMASS PosePrior" in _items(group)
    for name in ("PosePrior", "MPI_Limits"):  # current and former name of the range of motion subset
        _touch(str(datasets / "AMASS" / name / "03099" / "op2_poses.npz"))
        group, _ = cd.check_extra_3d(str(datasets), str(tmp_path), sampler)
        assert "AMASS PosePrior" not in _items(group)
        assert "range of motion) missing" not in _items(group)["AMASS converted"].detail
        os.remove(datasets / "AMASS" / name / "03099" / "op2_poses.npz")
        os.rmdir(datasets / "AMASS" / name / "03099")
        os.rmdir(datasets / "AMASS" / name)
