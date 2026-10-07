"""Training infrastructure (AMP / EMA / clipping / best epoch / resume / non-finite guard) on synthetic data."""
import math
import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.models.experiments.experiment_paths import ExperimentPaths
from pedrec.models.experiments.validation_set_model import ValidationSet
from pedrec.training import train_pedrec
from pedrec.training.experiments.checkpoint_selection import BestEpochTracker, relative_score
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.pedrec_stages import get_stage
from pedrec.training.experiments.train_stepper import TrainStepper, TrainingOptions
from pedrec.utils.torch_utils.torch_helper import split_no_wd_params


class SyntheticPoseDataset(Dataset):
    def __init__(self, num: int = 4):
        rng = np.random.default_rng(0)
        self.items = []
        for i in range(num):
            skeleton = np.concatenate((rng.random((26, 2)), np.ones((26, 3))), axis=1).astype(np.float32)
            skeleton_3d = np.concatenate((rng.random((26, 3)), np.ones((26, 3))), axis=1).astype(np.float32)
            orientation = np.concatenate((rng.random((2, 2)), np.ones((2, 3))), axis=1).astype(np.float32)
            self.items.append((torch.randn(3, 256, 192), {
                "skeleton": skeleton, "skeleton_3d": skeleton_3d, "orientation": orientation,
                "center": np.array([96, 128], dtype=np.float32), "scale": np.array([192, 256], dtype=np.float32),
                "rotation": 0.0, "env_position_2d": np.zeros(3, dtype=np.float32)}))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def test_stepper_skips_non_finite_losses():
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    stepper = TrainStepper(net, optimizer, None, torch.device("cpu"), TrainingOptions(ema_decay=0, max_bad_steps=2))
    before = net.weight.detach().clone()
    loss = net(torch.ones(1, 2)).sum() * float("nan")
    assert not stepper.step(loss)
    assert torch.equal(net.weight, before)
    assert stepper.stats.skipped_steps == 1
    assert stepper.step(net(torch.ones(1, 2)).sum())
    assert not torch.equal(net.weight, before)
    for _ in range(2):
        stepper.step(net(torch.ones(1, 2)).sum() * float("inf"))
    with pytest.raises(RuntimeError):
        stepper.step(net(torch.ones(1, 2)).sum() * float("inf"))


def test_stepper_gradient_accumulation_and_clipping():
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    stepper = TrainStepper(net, optimizer, None, torch.device("cpu"),
                           TrainingOptions(ema_decay=0, accumulate=2, grad_clip=0.01))
    assert not stepper.step(net(torch.ones(1, 2) * 100).sum())
    assert stepper.step(net(torch.ones(1, 2) * 100).sum())
    assert stepper.stats.clipped_steps == 1 and stepper.stats.optimizer_steps == 1


def test_relative_score_orientation():
    reference = {"COCO/pck@0.2": 80.0, "H36M/mpjpe": 60.0}
    assert relative_score({"COCO/pck@0.2": 80.0, "H36M/mpjpe": 60.0}, reference) == 1.0
    assert relative_score({"COCO/pck@0.2": 88.0, "H36M/mpjpe": 54.0}, reference) < 1.0
    assert relative_score({"COCO/pck@0.2": 70.0, "H36M/mpjpe": 70.0}, reference) > 1.0


def test_stage_rounds_ema_best_and_resume(tmp_path, monkeypatch):
    torch.manual_seed(0)
    device = torch.device("cpu")
    stage = get_stage("p2d3d_c_o_h36m_sim_mebow")
    paths = get_experiment_paths(str(tmp_path))
    paths.output_dir = str(tmp_path)
    description = train_pedrec.get_experiment_description(stage, paths, batch_size=2, num_workers=0)
    loader = DataLoader(SyntheticPoseDataset(), batch_size=2)
    val_sets = [ValidationSet(name="SYN", loader=DataLoader(SyntheticPoseDataset(), batch_size=2), val_set_cfg=None,
                              validate_2D=True, validate_3D=True, validate_orientation=True,
                              validate_pose_conf=True, validate_env_position=False)]
    monkeypatch.setattr(train_pedrec, "get_train_loader", lambda *a, **k: loader)
    monkeypatch.setattr(train_pedrec, "get_validation_sets", lambda *a, **k: val_sets)

    # predecessor checkpoint
    predecessor = train_pedrec.PedRecNetMTLWrapper(train_pedrec.PedRecNet(PedRecNet50Config()),
                                                   train_pedrec.PedRecNetLossHead(device))
    torch.save(predecessor.state_dict(), paths.get_stage_checkpoint_path(stage.init_from))
    options = TrainingOptions(amp="auto", ema_decay=0.9, grad_clip=5.0)
    net = train_pedrec.build_net(stage, description, device)
    train_pedrec.train_stage(stage, description, net, device, cycle_num=0, epochs_round_1=1, epochs_round_2=1,
                             options=options)
    for suffix in ["0_01.pth", "0.pth", "0_best.pth", "0_state.pth", "protocol.md"]:
        assert any(f.endswith(suffix) for f in os.listdir(tmp_path)), suffix
    final = torch.load(paths.get_stage_checkpoint_path(stage.name), weights_only=True)
    assert all(torch.isfinite(v).all() for v in final.values() if v.is_floating_point())

    # simulate an interruption after the first of two epochs of round 2 and resume
    description = train_pedrec.get_experiment_description(stage, paths, batch_size=2, num_workers=0)
    net = train_pedrec.build_net(stage, description, device)
    trained_epochs = []
    import pedrec.training.experiments.experiment_train_helper as helper
    original_train = helper.train

    def counting_train(*args, **kwargs):
        trained_epochs.append(1)
        return original_train(*args, **kwargs)
    monkeypatch.setattr(helper, "train", counting_train)

    class Interrupt(Exception):
        pass

    original_save = train_pedrec.StageState.save

    def save_and_interrupt(self, round_idx, epoch, *args, **kwargs):
        original_save(self, round_idx, epoch, *args, **kwargs)
        if round_idx == 1 and epoch == 0:
            raise Interrupt()
    monkeypatch.setattr(train_pedrec.StageState, "save", save_and_interrupt)
    with pytest.raises(Interrupt):
        train_pedrec.train_stage(stage, description, net, device, cycle_num=0, epochs_round_1=1, epochs_round_2=2,
                                 options=options)
    assert len(trained_epochs) == 2
    monkeypatch.setattr(train_pedrec.StageState, "save", original_save)

    description = train_pedrec.get_experiment_description(stage, paths, batch_size=2, num_workers=0)
    net = train_pedrec.build_net(stage, description, device)
    train_pedrec.train_stage(stage, description, net, device, cycle_num=0, epochs_round_1=1, epochs_round_2=2,
                             options=options, resume=True)
    assert len(trained_epochs) == 3  # only the missing epoch was trained
    state = torch.load(f"{paths.get_stage_file_base(stage.name)}_0_state.pth", weights_only=False)
    assert state["round"] == 1 and state["epoch"] == 1
    with pytest.raises(ValueError):  # changed schedule
        description = train_pedrec.get_experiment_description(stage, paths, batch_size=2, num_workers=0)
        train_pedrec.train_stage(stage, description, train_pedrec.build_net(stage, description, device), device,
                                 cycle_num=0, epochs_round_1=1, epochs_round_2=3, options=options, resume=True)


def test_ehpi3d_training_on_legacy_data(tmp_path, data_dir):
    from pedrec.training import train_ehpi3d
    from pedrec.training.experiments.ehpi3d_variants import get_variant
    paths = get_experiment_paths(str(tmp_path))
    paths.sim_c01_dir = data_dir
    paths.sim_c01_filename = "legacy_pedrec_gt_df.pkl"
    paths.sim_c01_results_filename = "legacy_pedrec_result_df.pkl"
    paths.ehpi3d_output_dir = str(tmp_path)
    checkpoint = train_ehpi3d.train_variant(get_variant("gt_pred_64frames"), paths, torch.device("cpu"), epochs=1,
                                            batch_size=8, num_workers=0, lr=1e-3,
                                            options=TrainingOptions(ema_decay=0.99))
    state = torch.load(checkpoint, weights_only=True)
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())


def test_v2_stage_initialized_from_v1_chain(tmp_path, monkeypatch):
    torch.manual_seed(0)
    device = torch.device("cpu")
    stage = get_stage("p2d3d_c_o_h36m_sim_mebow")
    paths = get_experiment_paths(str(tmp_path))
    paths.output_dir = str(tmp_path)
    # only the v1 predecessor checkpoint exists
    v1 = train_pedrec.PedRecNetMTLWrapper(train_pedrec.PedRecNet(PedRecNet50Config()),
                                          train_pedrec.PedRecNetLossHead(device))
    torch.save(v1.state_dict(), os.path.join(tmp_path, f"experiment_pedrec_{stage.init_from}_0.pth"))
    description = train_pedrec.get_experiment_description(stage, paths, batch_size=2, num_workers=0)
    assert description.coco_train_dataset_cfg.half_body_prob > 0
    loader = DataLoader(SyntheticPoseDataset(), batch_size=2)
    val_sets = [ValidationSet(name="SYN", loader=DataLoader(SyntheticPoseDataset(), batch_size=2), val_set_cfg=None,
                              validate_2D=True, validate_3D=True, validate_orientation=True,
                              validate_pose_conf=True, validate_env_position=False)]
    monkeypatch.setattr(train_pedrec, "get_train_loader", lambda *a, **k: loader)
    monkeypatch.setattr(train_pedrec, "get_validation_sets", lambda *a, **k: val_sets)
    net = train_pedrec.build_net(stage, description, device)
    train_pedrec.train_stage(stage, description, net, device, cycle_num=0, epochs_round_1=1, epochs_round_2=1,
                             options=TrainingOptions(ema_decay=0.9))
    checkpoint = os.path.join(tmp_path, f"experiment_pedrec_v2_{stage.name}_0.pth")
    assert os.path.isfile(checkpoint)
    assert "loss_head.log_vars" in torch.load(checkpoint, weights_only=True)
