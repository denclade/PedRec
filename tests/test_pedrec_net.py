"""PedRecNet v2 heads, losses, multi task weighting, stage initialization and UDP; random weights on the CPU."""
import math

import numpy as np
import pytest
import torch

from pedrec.configs.pedrec_net_config import PedRecNetConfig
from pedrec.inference import gpu_ops
from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet, PedRecNetLossHead
from pedrec.networks.net_pedrec.pedrec_net_factory import load_pedrec_net
from pedrec.networks.net_pedrec.pedrec_net_mtl_wrapper import PedRecNetMTLWrapper
from pedrec.networks.net_pedrec.pedrec_pose_conf_head_heatmap import heatmap_statistics
from pedrec.training.experiments.experiment_initializer import initialize_weights_with_same_name_and_shape
from pedrec.utils.augmentation_helper import get_affine_transforms, get_normalization_size
from pedrec.utils.torch_utils.loss_functions import BiternionLoss


def _targets(batch=2, joints=26, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "skeleton": torch.cat([torch.rand(batch, joints, 2, generator=g), torch.ones(batch, joints, 3)], dim=2),
        "skeleton_3d": torch.cat([torch.rand(batch, joints, 3, generator=g), torch.ones(batch, joints, 3)], dim=2),
        "orientation": torch.cat([torch.rand(batch, 2, 2, generator=g), torch.ones(batch, 2, 3)], dim=2),
    }


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    net = PedRecNet(PedRecNetConfig())
    net.init_weights()
    return net


def test_outputs_and_loss(net):
    outputs = net(torch.randn(2, 3, 256, 192))
    assert len(outputs) == 8
    assert outputs[0].shape == (2, 26, 3) and outputs[1].shape == (2, 26, 4) and outputs[2].shape == (2, 2, 2)
    assert outputs[5].shape == (2, 2, 2) and outputs[6].shape == (2, 2, 2) and outputs[7].shape == (2, 26)
    assert float(outputs[2].min()) >= 0 and float(outputs[2].max()) <= 1
    loss_head = PedRecNetLossHead(torch.device("cpu"))
    wrapper = PedRecNetMTLWrapper(net, loss_head)
    _, loss = wrapper(torch.randn(2, 3, 256, 192), _targets())
    assert torch.isfinite(loss)
    loss.backward()
    assert loss_head.log_vars.grad is not None
    assert net.head_conf.mlp[0].weight.grad is not None
    assert net.head_orientation.body_orientation.vectors.weight.grad is not None


def test_kendall_weighting_formula():
    head = PedRecNetLossHead(torch.device("cpu"))
    s = [0.5, -0.5, 1.0, 0.0]
    with torch.no_grad():
        head.log_vars.copy_(torch.tensor(s))
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5), torch.tensor(0.3)]
    head.task_losses = lambda outputs, targets: losses
    expected = sum((1.0 if i == 3 else 0.5) * math.exp(-s[i]) * losses[i].item() + 0.5 * s[i] for i in range(4))
    assert abs(head(None, None).item() - expected) < 1e-5


def test_weighting_skips_missing_and_disabled_tasks():
    head = PedRecNetLossHead(torch.device("cpu"), use_p3d_loss=False)
    losses = [torch.tensor(1.0), None, torch.tensor(float("nan")), torch.tensor(0.3)]
    head.task_losses = lambda outputs, targets: losses
    assert abs(head(None, None).item() - (0.5 * 1.0 + 0.3)) < 1e-6


def test_biternion_loss_is_continuous_at_wrap_around():
    loss = BiternionLoss(norm_weight=0)
    target = torch.zeros(1, 2, 5)
    target[:, :, 3:] = 1
    target[:, :, 0] = 0.5
    target[:, :, 1] = 359.0 / 360  # 359 degrees
    theta = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])  # 90 degrees
    phi = torch.tensor([[[math.cos(math.radians(1)), math.sin(math.radians(1))]] * 2])  # prediction: 1 degree
    assert loss(theta, phi, target).item() < 0.001  # only 2 degrees apart


def test_heatmap_statistics():
    peaked = torch.full((1, 1, 64, 48), -20.0)
    peaked[0, 0, 30, 20] = 20.0
    flat = torch.zeros(1, 1, 64, 48)
    stats = heatmap_statistics(torch.cat([peaked, flat], dim=1))
    assert stats[0, 0, 0] > 0.99 and stats[0, 0, 1] < 0.01 and stats[0, 0, 2] < 0.01  # peak, entropy, spread
    assert stats[0, 1, 0] < 0.01 and stats[0, 1, 1] > 0.99 and stats[0, 1, 2] > 0.3


def test_initialization_from_predecessor_checkpoint(tmp_path, net):
    """Matching weights are taken over, unknown keys and shape mismatches keep the initialization."""
    torch.manual_seed(1)
    source = PedRecNetMTLWrapper(PedRecNet(PedRecNetConfig()), PedRecNetLossHead(torch.device("cpu")))
    state = {k: v for k, v in source.state_dict().items() if not k.startswith("model.head_orientation.")}
    state["model.head_orientation.body_orientation.vectors.weight"] = torch.zeros(3, 3)  # shape mismatch
    state["model.unknown.weight"] = torch.zeros(1)
    path = str(tmp_path / "experiment_pedrec_v2_x_0.pth")
    torch.save(state, path)
    target = PedRecNetMTLWrapper(net, PedRecNetLossHead(torch.device("cpu")))
    orientation_before = net.head_orientation.body_orientation.vectors.weight.clone()
    initialize_weights_with_same_name_and_shape(target, path)
    first_backbone_param = next(net.backbone.parameters())
    assert torch.equal(first_backbone_param, next(source.model.backbone.parameters()))
    assert torch.equal(net.head_pose_3d.depth_map.weight, source.model.head_pose_3d.depth_map.weight)
    assert torch.equal(net.head_orientation.body_orientation.vectors.weight, orientation_before)


def test_load_pedrec_net_from_training_checkpoint(tmp_path, net):
    path = str(tmp_path / "experiment_pedrec_v2_x_0.pth")
    torch.save(PedRecNetMTLWrapper(net, PedRecNetLossHead(torch.device("cpu"))).state_dict(), path)
    loaded = load_pedrec_net(path)  # MTL checkpoint with the model. prefix
    x = torch.randn(1, 3, 256, 192)
    with torch.no_grad():
        assert torch.allclose(loaded(x)[0], net.eval()(x)[0], atol=1e-5)


def test_crop_normalize_roundtrip():
    """Dataset normalization (UDP) and the inference back-transformation must be inverse to each other."""
    input_size = ImageSize(192, 256)
    bb = np.array([500, 400, 150, 300, 1, 0], dtype=np.float32)
    trans, trans_inv = gpu_ops.get_crop_transforms([bb], input_size)
    joints = np.array([[480.0, 300.0], [530.0, 520.0], [500.0, 400.0]])
    crop = np.c_[joints, np.ones(3)] @ trans[0].T  # dataset: image -> crop pixels
    normalized = crop / get_normalization_size(input_size)
    scale = torch.from_numpy(get_normalization_size(input_size))
    back = gpu_ops.transform_coords_2d(torch.tensor(normalized, dtype=torch.float32)[None] * scale,
                                       torch.from_numpy(trans_inv))
    assert np.abs(back[0].numpy() - joints).max() < 1e-2
    # UDP: the bb center maps to the center pixel of the crop
    center = np.array([[500.0, 400.0, 1.0]]) @ get_affine_transforms(bb[:2], np.array([150, 200]), 0, input_size)[0].T
    assert np.allclose(center, [[95.5, 127.5]])
