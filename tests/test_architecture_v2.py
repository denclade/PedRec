"""Architecture variants (v2 heads, MTL weighting, backbones, UDP); all with random weights on the CPU."""
import dataclasses
import math
import os

import numpy as np
import pytest
import torch

from pedrec.configs.pedrec_net_config import PedRecArchConfig, get_arch_preset
from pedrec.inference import gpu_ops
from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet, PedRecNetLossHead
from pedrec.networks.net_pedrec.pedrec_net_factory import pedrec_config, save_arch, load_arch, load_pedrec_net
from pedrec.networks.net_pedrec.pedrec_net_mtl_wrapper import PedRecNetMTLWrapper
from pedrec.networks.net_pedrec.pedrec_pose_conf_head_heatmap import heatmap_statistics
from pedrec.training.experiments.experiment_initializer import initialize_weights_with_same_name_and_shape
from pedrec.utils.augmentation_helper import get_affine_transforms
from pedrec.utils.torch_utils.loss_functions import BiternionLoss


def _targets(batch=2, joints=26, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "skeleton": torch.cat([torch.rand(batch, joints, 2, generator=g), torch.ones(batch, joints, 3)], dim=2),
        "skeleton_3d": torch.cat([torch.rand(batch, joints, 3, generator=g), torch.ones(batch, joints, 3)], dim=2),
        "orientation": torch.cat([torch.rand(batch, 2, 2, generator=g), torch.ones(batch, 2, 3)], dim=2),
    }


@pytest.fixture(scope="module")
def v2_net():
    torch.manual_seed(0)
    net = PedRecNet(pedrec_config(get_arch_preset("v2")))
    net.init_weights()
    return net


def test_v2_outputs_and_loss(v2_net):
    outputs = v2_net(torch.randn(2, 3, 256, 192))
    assert len(outputs) == 8
    assert outputs[0].shape == (2, 26, 3) and outputs[1].shape == (2, 26, 4) and outputs[2].shape == (2, 2, 2)
    assert outputs[5].shape == (2, 2, 2) and outputs[6].shape == (2, 2, 2) and outputs[7].shape == (2, 26)
    assert float(outputs[2].min()) >= 0 and float(outputs[2].max()) <= 1
    loss_head = PedRecNetLossHead(torch.device("cpu"), weighting="kendall", orientation_head="biternion")
    wrapper = PedRecNetMTLWrapper(v2_net, loss_head)
    out, loss = wrapper(torch.randn(2, 3, 256, 192), _targets())
    assert torch.isfinite(loss)
    loss.backward()
    assert loss_head.log_vars.grad is not None
    assert v2_net.head_conf.mlp[0].weight.grad is not None
    assert v2_net.head_orientation.body_orientation.vectors.weight.grad is not None


def test_kendall_weighting_formula():
    head = PedRecNetLossHead(torch.device("cpu"), weighting="kendall")
    with torch.no_grad():
        head.log_vars.copy_(torch.tensor([0.5, -0.5, 1.0, 0.0]))
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5), torch.tensor(0.3)]
    head.task_losses = lambda outputs, targets: losses
    s = [0.5, -0.5, 1.0, 0.0]
    expected = sum((1.0 if i == 3 else 0.5) * math.exp(-s[i]) * losses[i].item() + 0.5 * s[i] for i in range(4))
    assert abs(head(None, None).item() - expected) < 1e-5


def test_original_weighting_unchanged():
    """The refactored loss head must give the same value as the original formula."""
    head = PedRecNetLossHead(torch.device("cpu"))
    with torch.no_grad():
        head.sigmas.copy_(torch.tensor([0.9, 1.1, 1.2, 0.8]))
    losses = [torch.tensor(1.0), torch.tensor(2.0), torch.tensor(0.5), torch.tensor(0.3)]
    head.task_losses = lambda outputs, targets: losses
    sig = [0.9, 1.1, 1.2, 0.8]
    expected = sum(losses[i].item() / (2 * sig[i] ** 2) for i in range(3)) + losses[3].item() / sig[3] ** 2 \
        + math.log(1 + np.prod([x ** 2 for x in sig]))
    assert abs(head(None, None).item() - expected) < 1e-5


def test_biternion_loss_is_continuous_at_wrap_around():
    loss = BiternionLoss(norm_weight=0)
    target = torch.zeros(1, 2, 5)
    target[:, :, 3:] = 1
    target[:, :, 0] = 0.5
    target[:, :, 1] = 359.0 / 360  # 359 degrees
    theta = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])  # 90 degrees
    phi = torch.tensor([[[math.cos(math.radians(1)), math.sin(math.radians(1))]] * 2])  # prediction: 1 degree
    value = loss(theta, phi, target).item()
    assert value < 0.001  # only 2 degrees apart


def test_heatmap_statistics():
    peaked = torch.full((1, 1, 64, 48), -20.0)
    peaked[0, 0, 30, 20] = 20.0
    flat = torch.zeros(1, 1, 64, 48)
    stats = heatmap_statistics(torch.cat([peaked, flat], dim=1))
    assert stats[0, 0, 0] > 0.99 and stats[0, 0, 1] < 0.01 and stats[0, 0, 2] < 0.01  # peak, entropy, spread
    assert stats[0, 1, 0] < 0.01 and stats[0, 1, 1] > 0.99 and stats[0, 1, 2] > 0.3


def test_v2_initialization_from_v1_checkpoint(tmp_path, v2_net):
    v1 = PedRecNetMTLWrapper(PedRecNet(pedrec_config()), PedRecNetLossHead(torch.device("cpu")))
    path = str(tmp_path / "v1.pth")
    torch.save(v1.state_dict(), path)
    v2 = PedRecNetMTLWrapper(PedRecNet(pedrec_config(get_arch_preset("v2"))),
                             PedRecNetLossHead(torch.device("cpu"), weighting="kendall", orientation_head="biternion"))
    initialize_weights_with_same_name_and_shape(v2, path)
    # backbone and pose heads are taken over, the new heads keep their initialization
    assert torch.equal(v2.model.feature_extractor.conv1.weight, v1.model.feature_extractor.conv1.weight)
    assert torch.equal(v2.model.head_pose_3d.depth_heatmap_layer.weight, v1.model.head_pose_3d.depth_heatmap_layer.weight)


def test_arch_sidecar_and_factory_loading(tmp_path, v2_net):
    path = str(tmp_path / "experiment_pedrec_v2_x_0.pth")
    wrapper = PedRecNetMTLWrapper(v2_net, PedRecNetLossHead(torch.device("cpu"), weighting="kendall",
                                                            orientation_head="biternion"))
    torch.save(wrapper.state_dict(), path)
    save_arch(path, v2_net.cfg.arch)
    assert load_arch(path).orientation_head == "biternion"
    net = load_pedrec_net(path)  # MTL checkpoint with model. prefix
    x = torch.randn(1, 3, 256, 192)
    with torch.no_grad():
        assert torch.allclose(net(x)[0], v2_net.eval()(x)[0], atol=1e-5)
    assert load_arch(str(tmp_path / "published.pth")) == PedRecArchConfig()  # no sidecar -> v1


@pytest.mark.parametrize("backbone,channels", [("timm:resnet18", 512), ("timm:vit_tiny_patch16_224", 192)])
def test_timm_backbones(backbone, channels):
    pytest.importorskip("timm")
    arch = dataclasses.replace(get_arch_preset("v2", backbone), backbone_pretrained=False)
    net = PedRecNet(pedrec_config(arch))
    net.init_weights()
    assert net.feature_extractor.out_channels == channels
    outputs = net(torch.randn(2, 3, 256, 192))
    assert outputs[3].shape[-2:] == (64, 48)  # heatmaps with 1/4 of the input resolution for stride 16 and 32
    assert outputs[0].shape == (2, 26, 3)


@pytest.mark.parametrize("udp", [False, True])
def test_crop_normalize_roundtrip(udp):
    """Dataset normalization and inference back-transformation must be inverse to each other (also with UDP)."""
    input_size = ImageSize(192, 256)
    bb = np.array([500, 400, 150, 300, 1, 0], dtype=np.float32)
    trans, trans_inv = gpu_ops.get_crop_transforms([bb], input_size, udp)
    joints = np.array([[480.0, 300.0], [530.0, 520.0], [500.0, 400.0]])
    crop = np.c_[joints, np.ones(3)] @ trans[0].T  # dataset: image -> crop pixels
    offset = 1 if udp else 0
    normalized = crop / np.array([input_size.width - offset, input_size.height - offset])
    scale = torch.tensor([input_size.width - offset, input_size.height - offset], dtype=torch.float32)
    back = gpu_ops.transform_coords_2d(torch.tensor(normalized, dtype=torch.float32)[None] * scale,
                                       torch.from_numpy(trans_inv))
    assert np.abs(back[0].numpy() - joints).max() < 1e-2
    if udp:  # the bb center maps to the center pixel of the crop
        center = np.array([[500.0, 400.0, 1.0]]) @ get_affine_transforms(bb[:2], np.array([150, 200]), 0, input_size,
                                                                        udp=True)[0].T
        assert np.allclose(center, [[95.5, 127.5]])
