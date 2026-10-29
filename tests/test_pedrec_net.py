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


def test_weighting_bounds_cover_small_pose_losses():
    """The optimal log variance log(L) of a converged normalized pose loss (~0.005) must lie inside the bounds."""
    head = PedRecNetLossHead(torch.device("cpu"), use_p3d_loss=False, use_orientation_loss=False,
                             use_conf_loss=False)
    with torch.no_grad():
        head.log_vars.copy_(torch.tensor([math.log(0.005), 0.0, 0.0, 0.0]))
    head.task_losses = lambda outputs, targets: [torch.tensor(0.005), None, None, None]
    expected = 0.5 * 1.0 + 0.5 * math.log(0.005)
    assert abs(head(None, None).item() - expected) < 1e-5
    head.reset_log_vars()
    assert torch.equal(head.log_vars.detach(), torch.zeros(4))


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


def _random_adapter():
    from pedrec.networks.net_pedrec.pedrec_convention_adapter import ConventionAdapter
    torch.manual_seed(0)
    adapter = ConventionAdapter()
    with torch.no_grad():
        adapter.offsets.normal_(0, 0.1)
    return adapter


def test_convention_adapter_identity_and_fixed_joints():
    from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT as J
    from pedrec.networks.net_pedrec.pedrec_convention_adapter import ConventionAdapter, NO_CONVENTION
    pose_2d, pose_3d = torch.rand(4, 26, 3), torch.rand(4, 26, 4)
    adapted_2d, adapted_3d = ConventionAdapter()(pose_2d, pose_3d, torch.tensor([0, 1, 2, 3]))
    assert torch.allclose(adapted_2d, pose_2d) and torch.allclose(adapted_3d, pose_3d)  # starts as identity
    adapter = _random_adapter()
    assert torch.allclose(adapter.matrices().sum(dim=-1), torch.ones(len(adapter.conventions), 26))
    adapted_2d, adapted_3d = adapter(pose_2d, pose_3d, torch.tensor([NO_CONVENTION, 2, 2, 5]))
    assert torch.allclose(adapted_2d[0], pose_2d[0]) and torch.allclose(adapted_3d[0], pose_3d[0])  # SIM
    for joint in (J.left_knee, J.right_ankle, J.left_elbow, J.right_wrist):  # anatomically defined: fixed
        assert torch.allclose(adapted_2d[:, joint.value], pose_2d[:, joint.value])
    assert not torch.allclose(adapted_2d[1, J.left_hip.value], pose_2d[1, J.left_hip.value])
    root = J.hip_center.value
    assert torch.allclose(adapted_3d[:, root, :3], pose_3d[:, root, :3], atol=1e-6)  # root kept
    assert torch.equal(adapted_3d[..., 3], pose_3d[..., 3]) and torch.equal(adapted_2d[..., 2], pose_2d[..., 2])


def test_convention_adapter_is_rotation_and_flip_equivariant():
    from pedrec.utils.skeleton_helper_3d import flip_lr_joints_3d
    adapter = _random_adapter()
    pose_3d = torch.rand(1, 26, 3)
    convention = torch.tensor([2])
    angle = 0.7
    rotation = torch.tensor([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
    _, adapted = adapter(pose_3d[..., :2], pose_3d, convention)
    _, adapted_rotated = adapter(pose_3d[..., :2], pose_3d @ rotation.T, convention)
    assert torch.allclose(adapted_rotated, adapted @ rotation.T, atol=1e-5)
    flipped = torch.from_numpy(flip_lr_joints_3d(pose_3d[0].numpy().copy()))[None]
    _, adapted_flipped = adapter(flipped[..., :2], flipped, convention)
    expected = torch.from_numpy(flip_lr_joints_3d(adapted[0].detach().numpy().copy()))[None]
    assert torch.allclose(adapted_flipped, expected, atol=1e-5)


def test_convention_adapter_learns_a_hip_convention():
    """Targets with the hips moved down by 15 % of the torso: only the adapter is trained and recovers the offset."""
    from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT as J
    from pedrec.networks.net_pedrec.pedrec_convention_adapter import ConventionAdapter, get_convention_index
    torch.manual_seed(0)
    pose = torch.rand(256, 26, 3)
    neck, hc = pose[:, J.neck.value], pose[:, J.hip_center.value]
    target = pose.clone()
    for hip in (J.left_hip, J.right_hip, J.hip_center):
        target[:, hip.value] -= 0.15 * (neck - hc)
    target -= target[:, J.hip_center.value:J.hip_center.value + 1] - hc[:, None]
    adapter = ConventionAdapter(reg_weight=0)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=0.01)
    convention = torch.full((256,), get_convention_index("h36m"))
    errors = []
    for _ in range(300):
        _, adapted = adapter(pose[..., :2], pose, convention)
        error = (adapted - target).abs().mean()
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        errors.append(error.item())
    assert errors[-1] < 0.1 * errors[0]
    description = adapter.describe()[get_convention_index("h36m")]
    assert description.startswith("h36m: ") and "hip_center" not in description  # root stays at the origin


def test_loss_head_without_adapter_ignores_conventions():
    head = PedRecNetLossHead(torch.device("cpu"))
    assert head.convention_adapter is None
    outputs = (torch.rand(2, 26, 3), torch.rand(2, 26, 4))
    assert head.adapt_outputs(outputs, {"convention": torch.tensor([2, 3])}) is outputs


def test_extract_network_from_checkpoint_with_convention_adapter(tmp_path, net):
    from pedrec.tools.networks.mtl_net_extractor import mtl_to_net
    path, output = str(tmp_path / "mtl.pth"), str(tmp_path / "net.pth")
    torch.save(PedRecNetMTLWrapper(net, PedRecNetLossHead(torch.device("cpu"), use_convention_adapter=True))
               .state_dict(), path)
    mtl_to_net(PedRecNetMTLWrapper(PedRecNet(PedRecNetConfig()), PedRecNetLossHead(torch.device("cpu"))), path, output)
    loaded = load_pedrec_net(path)  # inference loads only the network
    assert torch.equal(next(loaded.parameters()), next(net.parameters()))
    assert set(torch.load(output, weights_only=True)) == set(net.state_dict())


def test_convention_adapter_describes_net_offsets_in_body_coordinates():
    from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT as J
    from pedrec.networks.net_pedrec.pedrec_convention_adapter import ConventionAdapter, get_convention_index
    adapter = ConventionAdapter()
    h36m = get_convention_index("h36m")
    with torch.no_grad():  # wider hips: left_hip -= 0.1 (right_hip - left_hip), symmetrized to 0.05 per side
        adapter.offsets[h36m, J.left_hip.value, J.right_hip.value] = -0.1
    displacement = adapter.displacements()[h36m]
    assert displacement[J.left_hip.value, 0] > 10 and displacement[J.right_hip.value, 0] < -10  # mm, outwards
    assert "hip out +1.1 up +0.0 fwd +0.0 cm" in adapter.describe()[h36m]
    assert adapter.describe()[get_convention_index("fit3d")] == "fit3d: identity"


def test_joint_bias_body_shows_convention_offsets():
    from pedrec.evaluations.eval_np.eval_pose_3d import get_joint_bias_body
    from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT as J
    from pedrec.networks.net_pedrec.pedrec_convention_adapter import get_reference_pose
    rng = np.random.default_rng(0)
    reference = get_reference_pose().numpy()
    targets, preds = [], []
    for _ in range(8):  # random body rotations around the vertical axis
        angle = rng.uniform(0, 2 * np.pi)
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        pred = reference.copy()
        pred[J.left_hip.value, 0] += 20  # hips 2 cm wider
        pred[J.right_hip.value, 0] -= 20
        pred[J.head_upper.value, 1] -= 30  # head top 3 cm lower
        targets.append(np.concatenate([reference @ rotation.T, np.ones((26, 3))], axis=1))
        preds.append(np.concatenate([pred @ rotation.T, np.ones((26, 1))], axis=1))
    bias = get_joint_bias_body(np.stack(targets), np.stack(preds))
    assert np.allclose(bias[J.left_hip.value], [20, 0, 0], atol=0.5)
    assert np.allclose(bias[J.right_hip.value], [20, 0, 0], atol=0.5)  # outwards for both sides
    assert np.allclose(bias[J.head_upper.value], [0, -30, 0], atol=0.5)
    assert np.allclose(bias[J.left_knee.value], 0, atol=0.5)


def test_joint_bias_body_2d_in_percent_of_torso():
    from pedrec.evaluations.eval_np.eval_pose_2d import get_joint_bias_body_2d
    from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT as J
    from pedrec.networks.net_pedrec.pedrec_convention_adapter import get_reference_pose
    reference = get_reference_pose().numpy()[:, :2] * np.array([1, -1]) + 500  # image: y down
    for scale in (0.2, 0.5):  # independent of the person size in pixels
        target = reference * scale
        pred = target.copy()
        torso = np.linalg.norm((target[J.left_shoulder.value] + target[J.right_shoulder.value]) / 2 -
                               target[J.hip_center.value])
        pred[J.left_hip.value, 0] += 0.1 * torso  # hips 10 % torso wider (left side is image right here)
        pred[J.right_hip.value, 0] -= 0.1 * torso
        pred[J.head_upper.value, 1] += 0.05 * torso  # head top 5 % lower (image y down)
        labels = np.concatenate([target, np.ones((26, 3))], axis=1)[None]
        bias = get_joint_bias_body_2d(labels, np.concatenate([pred, np.ones((26, 3))], axis=1)[None])
        assert np.allclose(bias[J.left_hip.value], [10, 0], atol=0.1)
        assert np.allclose(bias[J.right_hip.value], [10, 0], atol=0.1)
        assert np.allclose(bias[J.head_upper.value], [0, -5], atol=0.1)


def test_joint_bias_is_a_median_robust_to_failures():
    from pedrec.evaluations.eval_np.eval_pose_3d import get_joint_bias_body
    from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT as J
    from pedrec.networks.net_pedrec.pedrec_convention_adapter import get_reference_pose
    reference = get_reference_pose().numpy()
    target = np.repeat(np.concatenate([reference, np.ones((26, 3))], axis=1)[None], 5, axis=0)
    pred = target.copy()
    pred[:, J.head_upper.value, 1] -= 20  # 2 cm lower in all frames
    pred[0, J.head_upper.value, 1] -= 500  # one failure
    assert np.isclose(get_joint_bias_body(target, pred)[J.head_upper.value, 1], -20)
