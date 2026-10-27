"""Dataset converters on synthetic mini datasets in the documented formats (no real data needed)."""
import json
import os
import pickle

import cv2
import numpy as np
import pytest

from pedrec.configs.dataset_configs import get_h36m_dataset_cfg_default
from pedrec.datasets.pedrec_dataset import PedRecDataset
from pedrec.datasets.pose_sequence_dataset import PoseSequenceDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.data_structures import ImageSize
from pedrec.utils.pandas_helper import read_pedrec_df

FRAMES = 12


def _person(frames: int, joints: int, seed: int = 0) -> np.ndarray:
    """Plausible standing person (meters, y down like a camera): N x joints x 3 around (0, 0, 4)."""
    rng = np.random.default_rng(seed)
    base = np.stack([rng.uniform(-0.3, 0.3, joints), np.linspace(-0.9, 0.9, joints), np.full(joints, 4.0)], axis=1)
    motion = np.linspace(0, 0.2, frames)[:, None, None] * np.array([1.0, 0, 0])
    return base[None] + motion + rng.normal(0, 0.01, (frames, joints, 3))


def _video(path: str, frames: int, width: int = 320, height: int = 240, fourcc: str = "mp4v"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), 25, (width, height))
    for i in range(frames):
        writer.write(np.full((height, width, 3), 40 + i, np.uint8))
    writer.release()


def _project(points_m: np.ndarray, f=300.0, c=(160.0, 120.0)) -> np.ndarray:
    return points_m[..., :2] / points_m[..., 2:3] * f + np.array(c)


def _check_outputs(image_df: str, seq_df: str, expected_images: int):
    df = read_pedrec_df(image_df)
    assert len(df) == expected_images
    root = os.path.dirname(image_df)
    dataset = PedRecDataset(root, os.path.basename(image_df), DatasetType.TRAIN, get_h36m_dataset_cfg_default(),
                            ImageSize(192, 256), None)
    img, labels = dataset[0]
    assert img.shape == (256, 192, 3)
    skeleton_3d = labels["skeleton_3d"]
    supported = skeleton_3d[:, 5] > 0
    assert supported.sum() >= 13
    from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT
    assert np.allclose(skeleton_3d[SKELETON_PEDREC_JOINT.hip_center.value, :3], 0.5)  # normalized, hip centered
    sequences = PoseSequenceDataset(seq_df, train=True)
    features, target, mask = sequences[len(sequences) - 1]
    assert features.shape[1:] == (26, 6) and mask.sum() >= 13
    return df


def _make_3dhp_train(root, scipy_io):
    seq_dir = root / "S1" / "Seq1"
    joints = _person(FRAMES, 28)
    annot2 = np.empty((14, 1), dtype=object)
    annot3 = np.empty((14, 1), dtype=object)
    for cam in range(14):
        annot2[cam, 0] = _project(joints).reshape(FRAMES, -1)
        annot3[cam, 0] = (joints * 1000).reshape(FRAMES, -1)
    os.makedirs(seq_dir)
    scipy_io.savemat(str(seq_dir / "annot.mat"), {"annot2": annot2, "annot3": annot3})
    _video(str(seq_dir / "imageSequence" / "video_0.avi"), FRAMES, fourcc="MJPG")


def test_mpi_inf_3dhp_downscaled_parallel(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    from pedrec.tools.datasets.convert_mpi_inf_3dhp import convert_train
    root = tmp_path / "3dhp"
    _make_3dhp_train(root, scipy_io)
    convert_train(str(root), str(root), image_step=5, subjects=[1, 2], cameras=[0], max_image_size=0, workers=0)
    full = read_pedrec_df(str(root / "mpi_inf_3dhp_train_pedrec.pkl"))
    # images of the first run (original size) are replaced, joints scaled accordingly
    convert_train(str(root), str(root), image_step=5, subjects=[1, 2], cameras=[0], max_image_size=160, workers=2)
    half = read_pedrec_df(str(root / "mpi_inf_3dhp_train_pedrec.pkl"))
    image = cv2.imread(str(root / "images" / "S1_Seq1_cam0" / "img_00001.jpg"))
    assert image.shape[:2] == (120, 160)
    for column in ("skeleton2d_left_shoulder_x", "skeleton2d_left_shoulder_y", "bb_center_x", "bb_width"):
        assert np.allclose(half[column], full[column] * 0.5, atol=1e-3), column
    assert list(half.columns) == list(full.columns) and len(half) == len(full) == 3
    _check_outputs(str(root / "mpi_inf_3dhp_train_pedrec.pkl"), str(root / "mpi_inf_3dhp_train_seq.pkl"), 3)


def test_mpi_inf_3dhp(tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    h5py = pytest.importorskip("h5py")
    from pedrec.tools.datasets.convert_mpi_inf_3dhp import convert_train, convert_test
    root = tmp_path / "3dhp"
    _make_3dhp_train(root, scipy_io)
    convert_train(str(root), str(root), image_step=5, subjects=[1], cameras=[0])
    _check_outputs(str(root / "mpi_inf_3dhp_train_pedrec.pkl"), str(root / "mpi_inf_3dhp_train_seq.pkl"), 3)

    ts_dir = root / "mpi_inf_3dhp_test_set" / "TS1"
    os.makedirs(ts_dir / "imageSequence")
    joints17 = _person(FRAMES, 17, seed=1)
    with h5py.File(ts_dir / "annot_data.mat", "w") as f:
        f["annot2"] = _project(joints17)[:, None]
        f["annot3"] = (joints17 * 1000)[:, None]
        f["valid_frame"] = np.ones(FRAMES)
    for i in range(FRAMES):
        cv2.imwrite(str(ts_dir / "imageSequence" / f"img_{i + 1:06d}.jpg"), np.full((240, 320, 3), 80, np.uint8))
    convert_test(str(root), str(root), image_step=4)
    df = _check_outputs(str(root / "mpi_inf_3dhp_val_pedrec.pkl"), str(root / "mpi_inf_3dhp_val_seq.pkl"), 3)
    assert set(df["img_dir"].astype(str)) == {os.path.join("images", "TS1")}


def test_fit3d(tmp_path):
    from pedrec.tools.datasets.convert_fit3d import convert
    root = tmp_path / "fit3d"
    subject = root / "train" / "s03"
    joints_cam = _person(FRAMES, 25)
    rotation = np.eye(3)
    translation = np.zeros((1, 3))
    os.makedirs(subject / "joints3d_25")
    with open(subject / "joints3d_25" / "squat.json", "w") as f:
        json.dump({"joints3d_25": joints_cam.tolist()}, f)  # world == camera (identity extrinsics)
    os.makedirs(subject / "camera_parameters" / "50591643")
    with open(subject / "camera_parameters" / "50591643" / "squat.json", "w") as f:
        json.dump({"extrinsics": {"R": rotation.tolist(), "T": translation.tolist()},
                   "intrinsics_wo_distortion": {"f": [300.0, 300.0], "c": [160.0, 120.0]}}, f)
    _video(str(subject / "videos" / "50591643" / "squat.mp4"), FRAMES)
    convert(str(root), str(root), image_step=5, val_subjects=[])
    df = _check_outputs(str(root / "fit3d_train_pedrec.pkl"), str(root / "fit3d_train_seq.pkl"), 3)
    # hip center (pelvis) is the origin of the 3D pose, 2D matches the projection
    x2d = df["skeleton2d_left_hip_x"].to_numpy()  # Fit3D: joint 1 is the left hip
    assert np.allclose(x2d, _project(joints_cam)[[0, 5, 10], 1, 0], atol=0.5)


def test_pedrec_dataset_subject_filter(tmp_path):
    """Held out subjects (SIM validation split) are excluded from training and the only ones validated."""
    test_fit3d(tmp_path)
    root = str(tmp_path / "fit3d")

    def length(**subjects):
        cfg = get_h36m_dataset_cfg_default()
        cfg.subsample = 1
        for key, value in subjects.items():
            setattr(cfg, key, value)
        return len(PedRecDataset(root, "fit3d_train_pedrec.pkl", DatasetType.VALIDATE, cfg, ImageSize(192, 256), None))
    assert length() == 3
    assert length(subjects_include=("s03",)) == 3 and length(subjects_include=("s09",)) == 0
    assert length(subjects_exclude=("s03",)) == 0
    from pedrec.configs.dataset_configs import get_sim_train_dataset_cfg_default, get_sim_val_dataset_cfg_default
    assert get_sim_train_dataset_cfg_default().subjects_exclude == get_sim_val_dataset_cfg_default().subjects_include


def test_fit3d_fix_leg_sides(tmp_path):
    import pandas as pd
    from pedrec.tools.datasets.convert_fit3d import fix_leg_sides, leg_side_agreement
    columns = {}
    for kind in ("skeleton2d", "skeleton3d"):
        for side, x in (("left", 1.0), ("right", -1.0)):
            for joint in ("shoulder", "hip", "knee", "ankle", "foot_end"):
                mirrored = -x if joint != "shoulder" else x
                for axis, value in (("x", mirrored), ("y", 0.0), ("z", 0.0), ("visible", 1.0)):
                    columns[f"{kind}_{side}_{joint}_{axis}"] = [value] * 3
    path = str(tmp_path / "fit3d_train_pedrec.pkl")
    pd.DataFrame(columns).to_pickle(path)
    assert leg_side_agreement(pd.read_pickle(path)) == -1
    assert fix_leg_sides(path)
    df = pd.read_pickle(path)
    assert leg_side_agreement(df) == 1 and (df["skeleton2d_left_knee_x"] == 1.0).all()
    assert not fix_leg_sides(path)  # idempotent


def test_aistpp(tmp_path):
    from pedrec.tools.datasets.convert_aistpp import convert
    root = tmp_path / "aistpp"
    seq = "gBR_sBM_cAll_d04_mBR0_ch01"
    joints = _person(FRAMES, 17) * 100  # cm
    os.makedirs(root / "keypoints3d")
    with open(root / "keypoints3d" / f"{seq}.pkl", "wb") as f:
        pickle.dump({"keypoints3d": joints, "keypoints3d_optim": joints}, f)
    os.makedirs(root / "cameras")
    with open(root / "cameras" / "mapping.txt", "w") as f:
        f.write(f"{seq} setting1\n")
    with open(root / "cameras" / "setting1.json", "w") as f:
        json.dump([{"name": "c01", "size": [320, 240], "matrix": [[300, 0, 160], [0, 300, 120], [0, 0, 1]],
                    "rotation": [0, 0, 0], "translation": [0, 0, 0], "distortions": [0, 0, 0, 0, 0]}], f)
    _video(str(root / "videos" / f"{seq.replace('cAll', 'c01')}.mp4"), FRAMES)
    convert(str(root), str(root), image_step=5, views=["c01"])
    df = _check_outputs(str(root / "aistpp_train_pedrec.pkl"), str(root / "aistpp_train_seq.pkl"), 3)
    assert df["skeleton3d_head_upper_supported"].astype(float).sum() == 0  # not provided by COCO keypoints
    assert df["skeleton3d_neck_supported"].astype(float).min() == 1  # derived from the shoulders


def test_amass_virtual_camera():
    from pedrec.tools.datasets.convert_amass import look_at_camera, virtual_camera
    rotation = look_at_camera(np.array([5.0, 0, 1.5]), np.array([0.0, 0, 1.0]))
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
    target_cam = rotation @ (np.array([0.0, 0, 1.0]) - np.array([5.0, 0, 1.5]))
    assert target_cam[2] > 0 and np.allclose(target_cam[:2], 0, atol=1e-9)  # target straight ahead
    up_cam = rotation @ np.array([0.0, 0, 1.0])
    assert up_cam[1] < 0  # world up is image up (negative y)
    rotation, position, focal = virtual_camera(np.zeros((10, 3)), np.random.default_rng(0))
    assert 4.0 <= np.linalg.norm(position[:2]) <= 8.0 and 1100 <= focal <= 1700


def _fake_smplh_model(path: str, hands: bool = True):
    """Random SMPL-H model with the real topology sizes (6890 vertices, 52 joints) in the smplx npz layout."""
    rng = np.random.default_rng(0)
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19] + \
              [20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35] + \
              [21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50]
    v_template = rng.uniform([-0.3, -0.2, -1.0], [0.3, 0.2, 0.8], (6890, 3)).astype(np.float32)
    regressor = np.zeros((52, 6890), np.float32)
    for j in range(52):
        regressor[j, rng.choice(6890, 10, replace=False)] = 0.1
    weights = np.zeros((6890, 52), np.float32)
    weights[np.arange(6890), rng.integers(0, 52, 6890)] = 1
    shapedirs = np.zeros((6890, 3, 16), np.float32)
    shapedirs[..., 15] = rng.normal(0, 0.05, (6890, 3))  # only the last (16th) shape component has an effect
    np.savez(path, v_template=v_template, shapedirs=shapedirs,
             posedirs=np.zeros((6890, 3, 51 * 9), np.float32), J_regressor=regressor, weights=weights,
             kintree_table=np.array([[4294967295 if p < 0 else p for p in parents], list(range(52))]),
             f=rng.integers(0, 6890, (100, 3)).astype(np.int64),
             **({"hands_componentsl": np.eye(45, dtype=np.float32), "hands_componentsr": np.eye(45, dtype=np.float32),
                 "hands_meanl": np.zeros(45, np.float32), "hands_meanr": np.zeros(45, np.float32)} if hands else {}))


@pytest.mark.parametrize("amass_layout", [False, True])
def test_amass(tmp_path, amass_layout):
    pytest.importorskip("smplx")
    from pedrec.tools.datasets.convert_amass import convert
    models = tmp_path / "body_models" / "smplh"
    for gender in ("male", "female", "neutral"):
        path = models / gender / "model.npz" if amass_layout else models / f"SMPLH_{gender.upper()}.npz"
        os.makedirs(path.parent, exist_ok=True)
        _fake_smplh_model(str(path), hands=not amass_layout)
    root = tmp_path / "amass"
    os.makedirs(root / "MPI_Limits" / "03099")
    frames = 240
    poses = np.zeros((frames, 156), np.float32)
    poses[:, 3 * 18 + 2] = np.linspace(0, 1.5, frames)  # raise an arm
    np.savez(root / "MPI_Limits" / "03099" / "op2_poses.npz", poses=poses,
             trans=np.zeros((frames, 3), np.float32), betas=np.zeros(16, np.float32), gender="male",
             mocap_framerate=120.0)
    from pedrec.tools.datasets.convert_amass import BodyModels
    body_models = BodyModels(str(tmp_path / "body_models"))
    data = dict(np.load(root / "MPI_Limits" / "03099" / "op2_poses.npz"))
    assert body_models._load("male").num_betas == 16  # not limited to 10 by smplx
    shaped = dict(data, betas=np.eye(16, dtype=np.float32)[15])
    assert not np.allclose(body_models.joints(data, 4, 10), body_models.joints(shaped, 4, 10))
    convert(str(root), str(root), str(tmp_path / "body_models"), None, val_every=1000)
    df = read_pedrec_df(str(root / "amass_train_seq.pkl"))
    assert len(df) == frames // 4  # 120 fps -> 30 fps
    sequences = PoseSequenceDataset(str(root / "amass_train_seq.pkl"), train=True)
    features, target, mask = sequences[len(sequences) - 1]
    assert features.shape[1:] == (26, 6) and mask.sum() == 26  # all joints supported


def test_skeleton_columns_are_read_by_name_not_position(tmp_path):
    """The first --fix-leg-sides only renamed the leg columns in place: the arrays must follow the names."""
    import pandas as pd
    from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
    from pedrec.tools.datasets.convert_fit3d import fix_leg_sides
    from pedrec.utils.pedrec_dataset_helper import get_filter_skeleton2d
    columns = {}
    for joint in SKELETON_PEDREC_JOINTS:
        for value in ("x", "y", "score", "visible", "supported"):
            columns[f"skeleton2d_{joint.name}_{value}"] = [float(joint.value)] * 2
        for value, x in (("x", 1.0 if joint.name.startswith("left") else -1.0), ("y", 0.0), ("z", 0.0)):
            columns[f"skeleton3d_{joint.name}_{value}"] = [x] * 2
    df = pd.DataFrame(columns)
    renamed = df.rename(columns={"skeleton2d_left_knee_x": "skeleton2d_right_knee_x",
                                 "skeleton2d_right_knee_x": "skeleton2d_left_knee_x"})
    skeleton = renamed[get_filter_skeleton2d(renamed)].to_numpy().reshape(2, 26, 5)
    assert skeleton[0, SKELETON_PEDREC_JOINTS[13].value, 0] == 14  # left_knee x now holds the right knee value
    assert skeleton[0, SKELETON_PEDREC_JOINTS[13].value, 1] == 13
    path = str(tmp_path / "fit3d_train_pedrec.pkl")
    renamed.to_pickle(path)
    assert fix_leg_sides(path)  # consistent sides, but the columns are put into joint order
    assert get_filter_skeleton2d(pd.read_pickle(path)) == [c for c in pd.read_pickle(path) if c.startswith("skeleton2d")]
    assert not fix_leg_sides(path)
