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
    group = cd.check_pedrec_training(paths, str(tmp_path / "datasets"), sampler)
    assert _items(group)["Human3.6m train images"].status == cd.MISSING
    assert "info h36m" in _items(group)["Human3.6m train videos"].fix  # no videos yet
    # only every 10th row is read by the training -> only these images are needed
    for i in range(1, n + 1, 10):
        _touch(os.path.join(paths.h36m_train_dir, "S1", "Images", "Walk 1.54138969", f"img_{i:05d}.jpg"))
    _touch(os.path.join(paths.h36m_train_dir, "S1", "Videos", "Walk 1.54138969.mp4"))
    group = cd.check_pedrec_training(paths, str(tmp_path / "datasets"), sampler)
    assert _items(group)["Human3.6m train images"].status == cd.OK
    assert "10/10" in _items(group)["Human3.6m train images"].detail
    os.remove(os.path.join(paths.h36m_train_dir, "S1", "Images", "Walk 1.54138969", "img_00011.jpg"))
    group = cd.check_pedrec_training(paths, str(tmp_path / "datasets"), sampler)
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
    # action recognition: dataframes, but no PedRecNet results yet
    for filename in (paths.sim_c01_filename, paths.sim_c01_val_filename):
        _df(os.path.join(paths.sim_c01_dir, filename), ["scene/cam1"], [1])
    assert cd.run(str(tmp_path), color=False) == 0
    out = capsys.readouterr().out
    assert "train:pedrec   ready" in out and "train:ehpi3d   not ready" in out
    assert "mise run download:datasets:pedrec --parts c01-results" in out
    for filename in (paths.sim_c01_results_filename, paths.sim_c01_val_results_filename):
        _touch(os.path.join(paths.sim_c01_dir, filename))
    cd.run(str(tmp_path), color=False)
    assert "train:ehpi3d   ready" in capsys.readouterr().out  # the SIM-C01 images are not needed


def test_merge_steps():
    assert cd.merge_steps(["a", "mise run download:datasets:pedrec --parts h36m", "a",
                           "mise run download:datasets:pedrec --parts rom"]) == \
        ["a", "mise run download:datasets:pedrec --parts h36m rom"]
    assert cd.merge_steps([f"mise run download:datasets:coco --parts {p}" for p in ("annotations", "train", "val")]) \
        == ["mise run download:datasets:coco"]

def test_h36m_missing_videos_per_subject(tmp_path):
    paths = get_experiment_paths(str(tmp_path))
    videos = ["S1/Images/Walking 1.54138969", "S5/Images/Directions 1.54138969", "S5/Images/Directions 2.54138969"]
    _df(os.path.join(paths.h36m_train_dir, paths.h36m_train_filename), videos, [1, 1, 1])
    for img_dir in videos[:2]:  # Directions 2 of S5 is missing (incomplete subject archive)
        subject, _, name = img_dir.split("/")
        _touch(os.path.join(paths.h36m_train_dir, subject, "Videos", f"{name}.mp4"))
    sampler = cd.ImageSampler(os.devnull, samples=10)
    group = cd.check_pedrec_training(paths, str(tmp_path / "datasets"), sampler)
    item = _items(group)["Human3.6m train videos"]
    assert item.status == cd.PARTIAL and "S5: 1/2 missing (e.g. Directions 2.54138969.mp4)" in item.detail
    assert "S1" not in item.detail and item.fix.endswith("data:h36m:images")
