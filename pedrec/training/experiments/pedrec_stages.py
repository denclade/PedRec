"""
Training stages of PedRecNet v2.

    imagenet (HGNetV2-B3 ImageNet weights)
      └─ p2d_c ── p2d3d_c_o

``p2d_c`` trains the neck and the 2D / joint confidence heads on COCO, Human3.6m and SIM (frozen backbone first, then
the full network). ``p2d3d_c_o`` adds the 3D pose and the body / head orientation (COCO with the MEBOW orientation
labels) and is the network used by the demo.

Naming: p2d = 2D pose, p3d = 3D pose, c = joint confidence, o = orientation.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

IMAGENET_INIT = "imagenet"


@dataclass(frozen=True)
class PedRecTrainingStage:
    name: str
    description: str
    init_from: str  # name of the predecessor stage or IMAGENET_INIT

    # datasets
    train_coco: bool = True
    train_h36m: bool = True
    train_sim: bool = True

    # loss terms / heads which are trained (2D pose is always trained)
    use_p3d_loss: bool = True
    use_conf_loss: bool = True
    use_orientation_loss: bool = True

    # COCO body orientation annotations (MEBOW)
    mebow_train: bool = False  # use MEBOW orientation labels for the COCO training set (disables rotation aug)
    mebow_val: bool = False  # use MEBOW orientation labels for the COCO validation set

    # schedule: round 1 trains everything but the backbone, round 2 the full network with the lr divided by
    # (backbone, heads)
    epochs_round_1: int = 2
    epochs_round_2: int = 20
    round_2_lr_divisors: Tuple[float, float] = (10.0, 2.0)

    # Fixed learning rate; None = run the LR range test to find one
    fixed_lr: Optional[float] = None

    sim_val_subsampling: int = 10
    batch_size: int = 64

    @property
    def train_sigmas(self) -> bool:
        """The MTL loss weights are only trained when more than the 2D pose loss is active."""
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
    PedRecTrainingStage(
        name="p2d_c", init_from=IMAGENET_INIT,
        description="2D pose + joint confidence on COCO + Human3.6m + SIM, backbone from ImageNet.",
        use_p3d_loss=False, use_orientation_loss=False,
        epochs_round_1=2, epochs_round_2=60, round_2_lr_divisors=(2.0, 1.0)),
    PedRecTrainingStage(
        name="p2d3d_c_o", init_from="p2d_c",
        description="Full network on COCO (MEBOW orientations) + Human3.6m + SIM. Used by the demo.",
        mebow_train=True, mebow_val=True,
        epochs_round_1=5, epochs_round_2=30, round_2_lr_divisors=(10.0, 2.0)),
]

STAGES: Dict[str, PedRecTrainingStage] = {stage.name: stage for stage in _STAGES}

DEFAULT_STAGE = "p2d3d_c_o"


def get_stage(name: str) -> PedRecTrainingStage:
    if name not in STAGES:
        raise KeyError(f"Unknown training stage '{name}'. Available: {', '.join(STAGES.keys())}")
    return STAGES[name]


def get_stage_chain(name: str) -> List[PedRecTrainingStage]:
    """The stage and all its predecessors in training order (first element has to be trained first)."""
    chain: List[PedRecTrainingStage] = []
    current = get_stage(name)
    while True:
        chain.insert(0, current)
        if current.init_from == IMAGENET_INIT:
            break
        current = get_stage(current.init_from)
    return chain


def format_stage_table() -> str:
    lines = [f"{'stage':12} {'init from':10} {'datasets':16} {'losses':22} {'epochs':8} description", "-" * 110]
    for stage in _STAGES:
        datasets = "+".join([n for n, used in (("coco", stage.train_coco), ("h36m", stage.train_h36m),
                                              ("sim", stage.train_sim)) if used])
        losses = "+".join([n for n, used in (("p2d", True), ("p3d", stage.use_p3d_loss), ("conf", stage.use_conf_loss),
                                            ("orient", stage.use_orientation_loss)) if used])
        epochs = f"{stage.epochs_round_1}+{stage.epochs_round_2}"
        lines.append(f"{stage.name:12} {stage.init_from:10} {datasets:16} {losses:22} {epochs:8} {stage.description}")
    return "\n".join(lines)
