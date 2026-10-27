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
without convention (-1) use the identity. After the mapping the 3D pose is moved back to the predicted root.
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
_SOURCES.update({
    J.left_foot_end: [J.left_ankle, J.left_knee], J.right_foot_end: [J.right_ankle, J.right_knee],
    J.left_hand_end: [J.left_wrist, J.left_elbow], J.right_hand_end: [J.right_wrist, J.right_elbow],
})


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
    def __init__(self, num_joints: int = 26, conventions: List[str] = None, reg_weight: float = 1e-3):
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

    def describe(self, top: int = 3, min_weight: float = 0.01) -> List[str]:
        """The largest learned offsets per convention, e.g. 'h36m: left_hip += 0.12 (right_hip - left_hip)'."""
        weights = self.offset_weights().detach().cpu()
        lines = []
        for c, name in enumerate(self.conventions):
            entries = []
            for k in range(weights.shape[1]):
                for i in range(weights.shape[2]):
                    if i != k and abs(weights[c, k, i]) >= min_weight:
                        entries.append((abs(float(weights[c, k, i])), k, i, float(weights[c, k, i])))
            entries.sort(reverse=True)
            terms = [f"{J(k).name} += {w:+.3f} ({J(i).name} - {J(k).name})" for _, k, i, w in entries[:top]]
            lines.append(f"{name}: " + ("; ".join(terms) if terms else "identity"))
        return lines
