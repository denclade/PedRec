"""Temporal 3D lifter: network, inference windows, sequence dataset, training and the result dataframe tool."""
import os
from collections import deque

import numpy as np
import pytest
import torch

from pedrec.datasets.pose_sequence_dataset import PoseSequenceDataset
from pedrec.inference.pipeline import TemporalLifter, RuntimeConfig, SKELETON_3D_RANGE
from pedrec.models.human import Human
from pedrec.networks.net_pedrec.pose_lifter import TemporalPoseLifter, lifter_features, WINDOW
from pedrec.training.experiments.train_stepper import TrainingOptions

DATA = os.path.join(os.path.dirname(__file__), "data")
GT = os.path.join(DATA, "legacy_pedrec_gt_df.pkl")
RESULTS = os.path.join(DATA, "legacy_pedrec_result_df.pkl")


def test_lifter_starts_as_identity():
    net = TemporalPoseLifter(26).eval()
    assert net.window == WINDOW
    features = torch.randn(3, WINDOW, 26, 6)
    with torch.no_grad():
        assert torch.allclose(net(features), features[:, -1, :, 3:6])
    assert sum(p.numel() for p in net.parameters()) < 1_000_000


def test_lifter_features():
    skeleton_2d = np.zeros((26, 3), np.float32)
    skeleton_2d[:, :2] = [150, 300]
    skeleton_2d[:, 2] = 0.8
    skeleton_3d = np.full((26, 4), 300.0, np.float32)
    features = lifter_features(skeleton_2d, skeleton_3d, np.array([100, 200, 50, 200, 1, 0]))
    assert features.shape == (26, 6)
    assert np.allclose(features[0], [0.25, 0.5, 0.8, 0.1, 0.1, 0.1])


def _human(uid, value):
    features = np.full((26, 6), value, np.float32)
    return Human(bb=[0, 0, 1, 1, 1, 0], skeleton_2d=np.ones((26, 3), np.float32),
                 skeleton_3d=np.ones((26, 4), np.float32), orientation=np.zeros((2, 2), np.float32), uid=uid,
                 lifter_features=features)


def test_inference_window_repeats_missing_frames():
    lifter = TemporalLifter("unused", 26, torch.device("cpu"), RuntimeConfig(), net=TemporalPoseLifter(26))
    history = deque([(1, np.full((26, 6), 1.0)), (2, np.full((26, 6), 2.0)), (5, np.full((26, 6), 5.0))])
    window = lifter._window(6, history)
    assert window.shape == (WINDOW, 26, 6)
    # frames before the first entry repeat it, missing frames (3, 4, 6) repeat the last known state
    assert list(window[-6:, 0, 0]) == [1.0, 2.0, 2.0, 2.0, 5.0, 5.0]
    assert np.all(window[:-6, 0, 0] == 1.0)


def test_inference_lifting_per_track_and_reset():
    lifter = TemporalLifter("unused", 26, torch.device("cpu"), RuntimeConfig(), net=TemporalPoseLifter(26))
    humans = [_human(1, 0.1), _human(2, 0.2)]
    lifter(1, humans)
    # untrained lifter = identity: the 3D pose is the per frame pose of the features (x 3000 mm)
    assert np.allclose(humans[0].skeleton_3d[:, :3], 0.1 * SKELETON_3D_RANGE, atol=1e-3)
    assert np.allclose(humans[1].skeleton_3d[:, 3], 1.0)  # confidence kept
    assert set(lifter.histories) == {1, 2}
    lifter.remove(1)
    assert set(lifter.histories) == {2}
    lifter.reset()
    assert lifter.histories == {}


@pytest.mark.parametrize("results", [None, RESULTS])
def test_sequence_dataset(results):
    train = PoseSequenceDataset(GT, results, train=True)
    features, target, mask = train[len(train) - 1]
    assert features.shape == (WINDOW, 26, 6) and target.shape == (26, 3) and mask.shape == (26,)
    val = PoseSequenceDataset(GT, results, train=False)
    assert np.array_equal(val[3][0], val[3][0])  # deterministic validation noise
    # the first frames of a sequence are padded with its first frame
    position = val.targets[3]
    rows = val._window_rows(position, 1, val.valid_gt)
    start = val.order[val.sequence_start[position]]
    assert np.all(rows[:-(position - val.sequence_start[position]) - 1] == start)
    assert rows[-1] == val.order[position]


def test_train_lifter_and_lift_results(tmp_path):
    from pedrec.training.train_lifter import train_lifter
    from pedrec.tools.resultwriters.lift_sim_c01_results import lift_results
    from pedrec.utils.pandas_helper import read_pedrec_df
    specs = [("fixture", GT, RESULTS)]
    weights = str(tmp_path / "lifter.pth")
    results = train_lifter(specs, specs, torch.device("cpu"), weights, epochs=2, batch_size=8, num_workers=0,
                           options=TrainingOptions(ema_decay=0.9))
    assert os.path.isfile(weights)
    assert np.isfinite(results["fixture"]["lifted_mm"]) and np.isfinite(results["fixture"]["per_frame_mm"])
    net = TemporalPoseLifter(26)
    net.load_state_dict(torch.load(weights, weights_only=True))
    output = str(tmp_path / "lifted.pkl")
    count = lift_results(GT, RESULTS, output, net, torch.device("cpu"))
    assert count > 0
    before, after = read_pedrec_df(RESULTS), read_pedrec_df(output)
    assert list(before.columns) == list(after.columns) and len(before) == len(after)
    assert np.allclose(before["skeleton2d_nose_x"], after["skeleton2d_nose_x"])  # only 3D x / y / z replaced
