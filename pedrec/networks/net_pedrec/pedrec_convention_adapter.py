"""
Joint convention adapter for the multi dataset training (training only, not part of the inference network).

The datasets place some joints differently (measured on the data, relative to the ankle -> shoulder height): the hips
of Human3.6m / MPI-INF-3DHP / Fit3D are ~10 cm higher and ~5 cm wider than in SIM, their shoulders narrower, the head
top of Human3.6m / Fit3D lower, AIST++ derives its neck from the shoulders. The network predicts one canonical
skeleton (SIM convention: all SIM sets are computed identically, as the SIM-C01 data of the lifter / action
recognition). For the loss of a sample from dataset d the prediction is mapped to the convention of d:

    p_d[k] = sum_i W_d[k, i] p[i],   sum_i W_d[k, i] = 1   <=>   p_d[k] = p[k] + sum_i W_d[k, i] (p[i] - p[k])

i.e. a per dataset offset built from bone vectors, so it rotates and scales with the body (affine combinations commute
with rotation, translation, scaling and the affine 2D crop transforms). Only definition dependent joints (torso, head,
hand / foot ends) may move, and only towards their neighbours; knees, ankles, elbows and wrists are anatomically well
defined (their spread within SIM is as large as between the datasets: body proportions, not conventions) and stay
fixed, so the adapter cannot learn body proportions. W is left / right symmetric (flip augmentation), SIM and samples
without convention (-1) use the identity.

The root (hip center) moves only along the spine (a different pelvis definition = all other joints shifted against the
root, as the 3D pose is moved back to the predicted root after the mapping). It must not take offsets from the hips:
the canonical root IS the hip mean, such weights cost nothing on correct poses but make the loss nearly invariant to a
shift of all other joints against the root (v2-2026-10-28: H36M root <- 0.86 * hip mean, the canonical skeleton drifted
~20 cm from its root on studio images, 206 mm canonical vs. 64 mm adapted MPJPE).
"""
from typing import List

import torch
import torch.nn as nn

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINT as J, SKELETON_PEDREC_LR_PAIRS

# datasets with an own joint convention; SIM (all sets) is the canonical one and has no entry
CONVENTIONS: List[str] = ["coco", "tud", "h36m", "mpi_inf_3dhp", "fit3d", "aistpp"]
NO_CONVENTION = -1  # canonical (SIM) or unknown: identity

_TORSO = [J.left_hip, J.right_hip, J.hip_center, J.spine_center, J.neck, J.left_shoulder, J.right_shoulder]
_HEAD = [J.nose, J.left_eye, J.right_eye, J.left_ear, J.right_ear, J.head_lower, J.head_upper, J.neck,
         J.left_shoulder, J.right_shoulder]
# adaptable joint -> joints its offset may be built from
_SOURCES = {joint: _TORSO for joint in _TORSO}
_SOURCES.update({joint: _HEAD for joint in [J.nose, J.left_eye, J.right_eye, J.left_ear, J.right_ear, J.head_lower,
                                            J.head_upper]})
_SOURCES[J.hip_center] = [J.spine_center, J.neck]  # not the hips, see module docstring
_SOURCES.update({
    J.left_foot_end: [J.left_ankle, J.left_knee], J.right_foot_end: [J.right_ankle, J.right_knee],
    J.left_hand_end: [J.left_wrist, J.left_elbow], J.right_hand_end: [J.right_wrist, J.right_elbow],
})


# Standing SIM pose in body coordinates (mm; x to the left side, y up, z forward; hip center at the origin): median of
# the upright frames of rt_rom_01b / rt_validate_3d, left / right symmetrized. Only used to report the learned offsets.
_REFERENCE_POSE_MM = {
    J.nose: (0, 709, 140), J.left_eye: (35, 745, 135), J.left_ear: (75, 750, 70), J.left_shoulder: (176, 572, 0),
    J.left_elbow: (268, 343, -19), J.left_wrist: (321, 118, 38), J.left_hip: (106, 0, 0), J.left_knee: (104, -434, -30),
    J.left_ankle: (90, -828, -85), J.hip_center: (0, 0, 0), J.spine_center: (0, 283, 16), J.neck: (0, 567, 34),
    J.head_lower: (0, 651, 59), J.head_upper: (0, 888, 76), J.left_foot_end: (122, -953, 94),
    J.left_hand_end: (296, 13, 66),
}


def get_reference_pose(num_joints: int = 26) -> torch.Tensor:
    """J x 3 standing reference pose in mm (body coordinates, see _REFERENCE_POSE_MM)."""
    pose = torch.zeros(num_joints, 3)
    for joint, (x, y, z) in _REFERENCE_POSE_MM.items():
        pose[joint.value] = torch.tensor([x, y, z], dtype=torch.float32)
    for left, right in SKELETON_PEDREC_LR_PAIRS:
        pose[right] = pose[left] * torch.tensor([-1.0, 1.0, 1.0])
    return pose


def get_convention_index(dataset_name: str) -> int:
    """Index of the dataset convention, NO_CONVENTION for SIM / datasets without an own convention."""
    return CONVENTIONS.index(dataset_name) if dataset_name in CONVENTIONS else NO_CONVENTION


def get_source_mask(num_joints: int) -> torch.Tensor:
    """J x J, 1 where joint k may take an offset from joint i (no diagonal: it is implied by the row sum)."""
    mask = torch.zeros(num_joints, num_joints)
    for joint, sources in _SOURCES.items():
        for source in sources:
            if source != joint:
                mask[joint.value, source.value] = 1
    return mask


def get_flip_permutation(num_joints: int) -> torch.Tensor:
    perm = list(range(num_joints))
    for left, right in SKELETON_PEDREC_LR_PAIRS:
        perm[left], perm[right] = right, left
    return torch.tensor(perm, dtype=torch.long)


class ConventionAdapter(nn.Module):
    def __init__(self, num_joints: int = 26, conventions: List[str] = None, reg_weight: float = 1e-6):
        super().__init__()
        self.conventions = list(conventions or CONVENTIONS)
        self.reg_weight = reg_weight
        self.offsets = nn.Parameter(torch.zeros(len(self.conventions), num_joints, num_joints))
        self.register_buffer("source_mask", get_source_mask(num_joints), persistent=False)
        self.register_buffer("flip_perm", get_flip_permutation(num_joints), persistent=False)
        self.register_buffer("eye", torch.eye(num_joints), persistent=False)

    def offset_weights(self) -> torch.Tensor:
        """C x J x J off diagonal weights (masked, left / right symmetric), rows of W - I sum to 0."""
        d = self.offsets * self.source_mask
        d = 0.5 * (d + d[:, self.flip_perm][:, :, self.flip_perm])
        return d - torch.diag_embed(d.sum(dim=-1))

    def matrices(self) -> torch.Tensor:
        """C x J x J mapping matrices W_d (rows sum to 1)."""
        return self.eye + self.offset_weights()

    def regularization(self) -> torch.Tensor:
        return self.reg_weight * torch.sum(self.offset_weights() ** 2)

    def forward(self, pose_2d: torch.Tensor, pose_3d: torch.Tensor, convention: torch.Tensor):
        """
        :param pose_2d: B x J x C2 (x, y, ...), :param pose_3d: B x J x C3 (x, y, z, ...), both normalized
        :param convention: B convention indices (NO_CONVENTION = identity)
        :return: pose_2d, pose_3d in the convention of each sample (further channels unchanged)
        """
        convention = convention.long().to(pose_2d.device)
        known = convention >= 0
        matrices = self.matrices()[convention.clamp(min=0)]
        matrices = torch.where(known[:, None, None], matrices, self.eye)
        pose_2d = torch.cat([matrices @ pose_2d[..., :2], pose_2d[..., 2:]], dim=-1)
        adapted_3d = matrices @ pose_3d[..., :3]
        # every dataset has its root (hip center) at the origin: keep the predicted root
        root = J.hip_center.value
        adapted_3d = adapted_3d - (adapted_3d[:, root:root + 1] - pose_3d[:, root:root + 1, :3])
        pose_3d = torch.cat([adapted_3d, pose_3d[..., 3:]], dim=-1)
        return pose_2d, pose_3d

    def displacements(self) -> torch.Tensor:
        """C x J x 3 net offsets (mm) of the standing reference pose in body coordinates (x left, y up, z forward)."""
        matrices = self.matrices().detach().cpu()
        reference = get_reference_pose(matrices.shape[1])
        adapted = matrices @ reference
        root = J.hip_center.value
        adapted = adapted - adapted[:, root:root + 1]
        return adapted - reference

    def describe(self, min_cm: float = 0.5) -> List[str]:
        """
        The learned net offsets per convention on the reference pose, left / right symmetric joints once (outward
        positive), e.g. 'h36m: hip out +2.3 up +6.1 fwd -0.4 cm'.
        """
        right_joints = {right for _, right in SKELETON_PEDREC_LR_PAIRS}
        displacements = self.displacements() / 10.0
        lines = []
        for c, name in enumerate(self.conventions):
            terms = []
            for k in range(displacements.shape[1]):
                if k in right_joints:
                    continue
                x, y, z = displacements[c, k].tolist()
                if max(abs(x), abs(y), abs(z)) < min_cm:
                    continue
                joint = J(k).name
                lateral = f"out {x:+.1f}" if joint.startswith("left_") else f"left {x:+.1f}"
                terms.append(f"{joint.replace('left_', '')} {lateral} up {y:+.1f} fwd {z:+.1f}")
            lines.append(f"{name}: " + (" | ".join(terms) + " cm" if terms else "identity"))
        return lines
