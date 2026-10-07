"""
Registry of the PedRecNet training stages.

The PedRecNet is trained in a chain of stages (see the experiment protocols in ``doc/diss_experiment_protocols``).
Each stage initializes from the checkpoint of its predecessor and adds datasets and/or loss terms:

    pose_resnet (COCO simple baselines)
      └─ p2d_coco_only ── p2d_c
           ├─ p2d_h36m ── p2d_h36m_sim
           │     └─ p2d3d_h36m ── p2d3d_h36m_sim
           │           └─ p2d3d_c_h36m ── p2d3d_c_h36m_sim ── p2d3d_c_o_h36m_sim ── p2d3d_c_o_h36m_sim_mebow ── p2d3d_c_o_h36m_sim_mebow_tud
           │                 ├─ p2d3d_c_o_h36m_mebow                       └─ p2d3d_c_o_h36m_sim_tud
           │                 └─ p2d3d_c_o_h36m_tud
           └─ p2d_sim ── p2d3d_sim ── p2d3d_c_sim ── p2d3d_c_o_sim

Naming: p2d = 2D pose, p3d = 3D pose, c = joint confidence, o = orientation, h36m / sim / tud / mebow = datasets.
The final network used in the demo is ``p2d3d_c_o_h36m_sim_mebow``.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional

POSE_RESNET_INIT = "pose_resnet"

# Learning rate found by the LR range test for p2d_h36m, reused by most later stages (see original experiments).
DEFAULT_FIXED_LR = 2.08e-03


@dataclass(frozen=True)
class PedRecTrainingStage:
    name: str
    description: str
    init_from: str  # name of the predecessor stage or POSE_RESNET_INIT

    # datasets
    train_coco: bool = True
    train_h36m: bool = True
    train_sim: bool = True
    train_tud: bool = False
    val_tud: bool = False

    # loss terms / heads which are trained (2D pose is always trained)
    use_p3d_loss: bool = True
    use_conf_loss: bool = True
    use_orientation_loss: bool = True

    # COCO body orientation annotations (MEBOW)
    mebow_train: bool = False  # use MEBOW orientation labels for the COCO training set (disables rotation aug)
    mebow_val: bool = False  # use MEBOW orientation labels for the COCO validation set

    # Re-initialize the orientation head although it exists in the predecessor checkpoint
    reinit_orientation_head: bool = False

    # Fixed learning rate; None = run the LR range test to find one
    fixed_lr: Optional[float] = None

    sim_val_subsampling: int = 10
    batch_size: int = 48

    @property
    def train_sigmas(self) -> bool:
        """The MTL loss weights (sigmas) are only trained when more than the 2D pose loss is active."""
        return self.use_p3d_loss or self.use_conf_loss or self.use_orientation_loss

    @property
    def validate_3d(self) -> bool:
        return self.use_p3d_loss

    @property
    def validate_orientation(self) -> bool:
        return self.use_orientation_loss

    @property
    def validate_joint_conf(self) -> bool:
        return self.use_conf_loss


_STAGES: List[PedRecTrainingStage] = [
    # ---------------------------------------------------------------- 2D pose
    PedRecTrainingStage(
        name="p2d_coco_only", init_from=POSE_RESNET_INIT,
        description="2D pose on COCO only, initialized from the pose-resnet simple baseline weights.",
        train_h36m=False, train_sim=False,
        use_p3d_loss=False, use_conf_loss=False, use_orientation_loss=False),
    PedRecTrainingStage(
        name="p2d_c", init_from="p2d_coco_only",
        description="2D pose + joint confidence on COCO only.",
        train_h36m=False, train_sim=False,
        use_p3d_loss=False, use_orientation_loss=False),
    PedRecTrainingStage(
        name="p2d_h36m", init_from="p2d_coco_only",
        description="2D pose on COCO + Human3.6m.",
        train_sim=False,
        use_p3d_loss=False, use_conf_loss=False, use_orientation_loss=False),
    PedRecTrainingStage(
        name="p2d_sim", init_from="p2d_coco_only",
        description="2D pose on COCO + SIM (ROMb / RT3DValidate).",
        train_h36m=False,
        use_p3d_loss=False, use_conf_loss=False, use_orientation_loss=False),
    PedRecTrainingStage(
        name="p2d_h36m_sim", init_from="p2d_h36m",
        description="2D pose on COCO + Human3.6m + SIM.",
        use_p3d_loss=False, use_conf_loss=False, use_orientation_loss=False, fixed_lr=DEFAULT_FIXED_LR),
    # ---------------------------------------------------------------- 2D + 3D pose
    PedRecTrainingStage(
        name="p2d3d_h36m", init_from="p2d_h36m",
        description="2D + 3D pose on COCO + Human3.6m.",
        train_sim=False,
        use_conf_loss=False, use_orientation_loss=False, fixed_lr=DEFAULT_FIXED_LR),
    PedRecTrainingStage(
        name="p2d3d_sim", init_from="p2d_sim",
        description="2D + 3D pose on COCO + SIM.",
        train_h36m=False,
        use_conf_loss=False, use_orientation_loss=False, fixed_lr=DEFAULT_FIXED_LR),
    PedRecTrainingStage(
        name="p2d3d_h36m_sim", init_from="p2d3d_h36m",
        description="2D + 3D pose on COCO + Human3.6m + SIM.",
        use_conf_loss=False, use_orientation_loss=False, fixed_lr=DEFAULT_FIXED_LR),
    # ---------------------------------------------------------------- + joint confidence
    PedRecTrainingStage(
        name="p2d3d_c_h36m", init_from="p2d3d_h36m",
        description="2D + 3D pose + joint confidence on COCO + Human3.6m.",
        train_sim=False,
        use_orientation_loss=False),
    PedRecTrainingStage(
        name="p2d3d_c_sim", init_from="p2d3d_sim",
        description="2D + 3D pose + joint confidence on COCO + SIM.",
        train_h36m=False,
        use_orientation_loss=False),
    PedRecTrainingStage(
        name="p2d3d_c_h36m_sim", init_from="p2d3d_c_h36m",
        description="2D + 3D pose + joint confidence on COCO + Human3.6m + SIM.",
        use_orientation_loss=False, fixed_lr=DEFAULT_FIXED_LR),
    # ---------------------------------------------------------------- + orientation
    PedRecTrainingStage(
        name="p2d3d_c_o_h36m_mebow", init_from="p2d3d_c_h36m",
        description="Full network on COCO (MEBOW orientations) + Human3.6m.",
        train_sim=False,
        mebow_train=True, mebow_val=True, fixed_lr=DEFAULT_FIXED_LR),
    PedRecTrainingStage(
        name="p2d3d_c_o_sim", init_from="p2d3d_c_sim",
        description="Full network on COCO + SIM (orientation labels from SIM only).",
        train_h36m=False,
        mebow_val=True, fixed_lr=DEFAULT_FIXED_LR),
    PedRecTrainingStage(
        name="p2d3d_c_o_h36m_sim", init_from="p2d3d_c_h36m_sim",
        description="Full network on COCO + Human3.6m + SIM (orientation labels from SIM only).",
        mebow_val=True, fixed_lr=DEFAULT_FIXED_LR),
    PedRecTrainingStage(
        name="p2d3d_c_o_h36m_sim_mebow", init_from="p2d3d_c_o_h36m_sim",
        description="Full network on COCO (MEBOW orientations) + Human3.6m + SIM. This is the published model.",
        mebow_train=True, mebow_val=True, reinit_orientation_head=True, fixed_lr=DEFAULT_FIXED_LR),
    # ---------------------------------------------------------------- + TUD
    PedRecTrainingStage(
        name="p2d3d_c_o_h36m_tud", init_from="p2d3d_c_h36m",
        description="Full network on COCO + Human3.6m + TUD orientations.",
        train_sim=False, train_tud=True, val_tud=True,
        mebow_val=True, fixed_lr=DEFAULT_FIXED_LR),
    PedRecTrainingStage(
        name="p2d3d_c_o_h36m_sim_tud", init_from="p2d3d_c_o_h36m_sim",
        description="Full network on COCO + Human3.6m + SIM + TUD orientations.",
        train_tud=True, val_tud=True,
        mebow_val=True, reinit_orientation_head=True, fixed_lr=DEFAULT_FIXED_LR),
    PedRecTrainingStage(
        name="p2d3d_c_o_h36m_sim_mebow_tud", init_from="p2d3d_c_o_h36m_sim_mebow",
        description="Full network on COCO (MEBOW) + Human3.6m + SIM + TUD orientations.",
        train_tud=True, val_tud=True,
        mebow_train=True, mebow_val=True, fixed_lr=DEFAULT_FIXED_LR),
]

STAGES: Dict[str, PedRecTrainingStage] = {stage.name: stage for stage in _STAGES}

DEFAULT_STAGE = "p2d3d_c_o_h36m_sim_mebow"


def get_stage(name: str) -> PedRecTrainingStage:
    key = name[len("experiment_pedrec_"):] if name.startswith("experiment_pedrec_") else name
    if key not in STAGES:
        raise KeyError(f"Unknown training stage '{name}'. Available: {', '.join(STAGES.keys())}")
    return STAGES[key]


def get_stage_chain(name: str) -> List[PedRecTrainingStage]:
    """
    Returns the stage and all its predecessors in training order (first element has to be trained first).
    """
    chain: List[PedRecTrainingStage] = []
    current = get_stage(name)
    while True:
        chain.insert(0, current)
        if current.init_from == POSE_RESNET_INIT:
            break
        current = get_stage(current.init_from)
    return chain


def format_stage_table() -> str:
    lines = [f"{'stage':32} {'init from':28} {'datasets':22} {'losses':18} description",
             "-" * 120]
    for stage in _STAGES:
        datasets = "+".join([n for n, used in (("coco", stage.train_coco), ("h36m", stage.train_h36m),
                                              ("sim", stage.train_sim), ("tud", stage.train_tud)) if used])
        losses = "+".join([n for n, used in (("p2d", True), ("p3d", stage.use_p3d_loss), ("conf", stage.use_conf_loss),
                                            ("orient", stage.use_orientation_loss)) if used])
        lines.append(f"{stage.name:32} {stage.init_from:28} {datasets:22} {losses:18} {stage.description}")
    return "\n".join(lines)
