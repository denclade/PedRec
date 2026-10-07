"""
Unified training entry point for the PedRecNet.

Replaces the former ``experiment_pedrec_*.py`` scripts: every training stage of the original experiment chain is
described in ``pedrec.training.experiments.pedrec_stages`` and can be trained via

    python pedrec/training/train_pedrec.py --stage p2d3d_c_o_h36m_sim_mebow

Run ``--list`` to see all stages and their dependencies. Dataset and checkpoint paths are derived from the data root
(``--data-dir`` or the ``PEDREC_DATA_DIR`` environment variable, default ``data``).
"""
import argparse
import logging
import os
import random
import sys
from typing import Optional

sys.path.append(".")

import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import OneCycleLR

from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.models.experiments.experiment_description import ExperimentDescription
from pedrec.models.experiments.experiment_round_description import ExperimentRoundDescription
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet, PedRecNetLossHead
from pedrec.networks.net_pedrec.pedrec_net_mtl_wrapper import PedRecNetMTLWrapper
from pedrec.training.experiments.experiment_dataset_helper import get_validation_sets, get_train_loader
from pedrec.training.experiments.experiment_initializer import initialize_weights_with_same_name_and_shape, \
    initialize_pose_resnet
from pedrec.training.experiments.experiment_log_helper import get_experiment_protocol, save_log
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.checkpoint_selection import BestEpochTracker
from pedrec.training.experiments.experiment_train_helper import init_experiment, train_round, get_outputs_loss_mtl
from pedrec.training.experiments.train_stepper import TrainingOptions, TrainStepper, steps_per_epoch
from pedrec.training.experiments.pedrec_stages import PedRecTrainingStage, get_stage, get_stage_chain, \
    format_stage_table, POSE_RESNET_INIT, STAGES, DEFAULT_STAGE
from pedrec.utils.torch_utils.mtl_lr_finder import MTLLRFinder
from pedrec.utils.torch_utils.torch_helper import get_device, split_no_wd_params

logger = logging.getLogger(__name__)

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

OPTIMIZER_PARAMS = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-2,
    "amsgrad": False
}


def get_preds_mtl(outputs):
    return {
        "skeleton": outputs[0].cpu().detach().numpy(),
        "skeleton_3d": outputs[1].cpu().detach().numpy(),
        "orientation": outputs[2].cpu().detach().numpy()
    }


def get_experiment_description(stage: PedRecTrainingStage, experiment_paths, batch_size: int,
                               num_workers: int) -> ExperimentDescription:
    description = ExperimentDescription(
        net_name="PedRecNet",
        experiment_name=stage.experiment_name,
        initialization_notes=f"Initialized from {stage.init_from}",
        experiment_paths=experiment_paths,
        net_cfg=PedRecNet50Config(),
        use_train_coco=stage.train_coco,
        use_train_h36m=stage.train_h36m,
        use_train_sim=stage.train_sim,
        use_train_tud=stage.train_tud,
        use_val_tud=stage.val_tud,
        validate_3d_sim=stage.validate_3d,
        validate_3d_h36m=stage.validate_3d,
        validate_orientation_sim=stage.validate_orientation,
        validate_orientation_coco=stage.validate_orientation,
        validate_orientation_tud=stage.validate_orientation and stage.val_tud,
        validate_joint_conf_sim=stage.validate_joint_conf,
        validate_joint_conf_coco=stage.validate_joint_conf,
        validate_joint_conf_h36m=stage.validate_joint_conf,
        sim_val_subsampling=stage.sim_val_subsampling,
        batch_size=batch_size,
        batch_size_validate=batch_size,
        num_workers=num_workers,
    )
    description.coco_val_dataset_cfg.use_mebow_orientation = stage.mebow_val
    description.coco_train_dataset_cfg.use_mebow_orientation = stage.mebow_train
    if stage.mebow_train:
        # orientation labels are not rotation invariant
        description.coco_train_dataset_cfg.rotation_factor = 0
    return description


def build_net(stage: PedRecTrainingStage, description: ExperimentDescription, device: torch.device,
              init_weights_path: str = None) -> PedRecNetMTLWrapper:
    net = PedRecNet(description.net_cfg)
    net.init_weights()
    loss_head = PedRecNetLossHead(device,
                                  use_p3d_loss=stage.use_p3d_loss,
                                  use_orientation_loss=stage.use_orientation_loss,
                                  use_conf_loss=stage.use_conf_loss)
    net = PedRecNetMTLWrapper(net, loss_head)

    paths = description.experiment_paths
    if init_weights_path is not None:
        logger.info(f"Initializing from {init_weights_path}")
        initialize_weights_with_same_name_and_shape(net, init_weights_path)
    elif stage.init_from == POSE_RESNET_INIT:
        logger.info(f"Initializing from pose-resnet weights {paths.pose_resnet_weights_path}")
        initialize_pose_resnet(net, paths.pose_resnet_weights_path)
    else:
        predecessor = paths.get_stage_checkpoint_path(stage.init_from)
        if not os.path.isfile(predecessor):
            raise FileNotFoundError(
                f"Checkpoint of predecessor stage '{stage.init_from}' not found: {predecessor}. "
                f"Train it first (python pedrec/training/train_pedrec.py --stage {stage.init_from}) or "
                f"download it (see README) or pass --init-weights.")
        logger.info(f"Initializing from predecessor stage checkpoint {predecessor}")
        initialize_weights_with_same_name_and_shape(net, predecessor)
    if stage.reinit_orientation_head:
        net.model.head_orientation.init_weights()
    net.to(device)

    description.net_layer_names = [
        "net.model.feature_extractor",
        "net.model.conv_transpose_shared",
        "net.model.head_pose_2d",
        "net.model.head_pose_3d",
        "net.model.head_orientation",
        "net.model.head_conf"
    ]
    description.net_layers = [
        net.model.feature_extractor,
        net.model.conv_transpose_shared,
        net.model.head_pose_2d,
        net.model.head_pose_3d,
        net.model.head_orientation,
        net.model.head_conf
    ]
    return net


def find_learning_rate(net, params, train_loader, device, output_path: str) -> float:
    """
    Runs the LR range test and stores the loss / learning rate plot as PNG (no interactive window).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    optimizer = torch.optim.AdamW(params, lr=1e-7, weight_decay=1e-2)
    lr_finder = MTLLRFinder(optimizer, net, device=device)
    lr_finder.range_test(train_loader, end_lr=0.05, num_iter=100)
    fig, ax = plt.subplots()
    result = lr_finder.plot(ax=ax)
    lr_finder.reset()
    if isinstance(result, tuple):
        _, suggested_lr = result
    else:
        raise RuntimeError("LR range test did not return a suggestion (loss did not decrease). "
                           "Pass a learning rate explicitly via --lr.")
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"LR range test suggested lr={suggested_lr:.2e} (plot: {output_path})")
    return suggested_lr


def get_max_lrs(stage: PedRecTrainingStage, base_lr: float, feature_extractor_divisor: float = 1.0,
                head_divisor: float = 1.0):
    """
    One max lr per parameter group (two groups per layer: with / without weight decay, plus the sigmas).
    Heads whose loss is disabled get lr 0.
    """
    fe = base_lr / feature_extractor_divisor
    head = base_lr / head_divisor
    return [
        fe, fe,  # feature extractor
        head, head,  # conv_transpose_shared
        head, head,  # pose 2d
        head if stage.use_p3d_loss else 0, head if stage.use_p3d_loss else 0,  # pose 3d
        head if stage.use_orientation_loss else 0, head if stage.use_orientation_loss else 0,  # orientation
        head if stage.use_conf_loss else 0, head if stage.use_conf_loss else 0,  # conf
        head if stage.train_sigmas else 0,  # sigmas
    ]


class StageState:
    """
    Everything needed to continue an interrupted stage training (``--resume``) plus the best epoch bookkeeping.
    Written after every epoch to ``experiment_pedrec_<stage>_<cycle>_state.pth``.
    """

    def __init__(self, path: str):
        self.path = path
        self.best = BestEpochTracker()
        self.loaded: Optional[dict] = None

    def load(self) -> bool:
        if not os.path.isfile(self.path):
            return False
        # own file incl. optimizer / RNG states -> full pickle
        self.loaded = torch.load(self.path, map_location="cpu", weights_only=False)
        self.best.load_state_dict(self.loaded.get("best", {}))
        return True

    def save(self, round_idx: int, epoch: int, net, ema, round_description: ExperimentRoundDescription, stepper,
             suggested_lr: float):
        state = {
            "round": round_idx, "epoch": epoch, "suggested_lr": suggested_lr,
            "model": net.state_dict(), "ema": ema.state_dict() if ema is not None else None,
            "optimizer": round_description.optimizer.state_dict(),
            "scheduler": round_description.scheduler.state_dict(),
            "stepper": stepper.state_dict(), "best": self.best.state_dict(),
            "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        tmp = self.path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.path)


def _final_weights(net, ema):
    return (ema.module if ema is not None else net).state_dict()


def run_round(round_idx: int, net, description: ExperimentDescription, params, max_lrs, epochs: int, frozen_layers,
              train_loader, validation_sets, device, checkpoint_path: str, options: TrainingOptions, ema,
              stage_state: StageState, best_checkpoint_path: str):
    optimizer = torch.optim.AdamW(params, **OPTIMIZER_PARAMS)
    scheduler = OneCycleLR(optimizer, max_lr=max_lrs, epochs=epochs,
                           steps_per_epoch=steps_per_epoch(len(train_loader), options.accumulate))
    round_description = ExperimentRoundDescription(
        num_epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        frozen_layers=frozen_layers,
        max_lrs=max_lrs,
        optimizer_parameters=OPTIMIZER_PARAMS
    )
    description.experiment_rounds.append(round_description)

    start_epoch, stepper_state = 0, None
    resumed = stage_state.loaded
    if resumed is not None and resumed["round"] == round_idx:
        if resumed["scheduler"].get("total_steps") != scheduler.total_steps:
            raise ValueError("The interrupted training used a different number of epochs / batches per epoch. "
                             "Resume with the same --epochs-round-*, --batch-size and --accumulate.")
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        stepper_state = resumed.get("stepper")
        start_epoch = resumed["epoch"] + 1
        logger.info(f"Resuming round {round_idx + 1} at epoch {start_epoch + 1}/{epochs}")

    def on_epoch_end(epoch, eval_net, results, stepper):
        name = f"round {round_idx + 1} epoch {epoch + 1}"
        if stage_state.best.update(results.validation_results, name):
            torch.save(eval_net.state_dict(), best_checkpoint_path)
            logger.info(f"New best epoch ({name}, relative score {stage_state.best.best_score:.4f}): "
                        f"{best_checkpoint_path}")
        stage_state.save(round_idx, epoch, net, ema, round_description, stepper, description.suggested_lr)

    round_description.validation_results = train_round(
        net, description, round_description, train_loader, validation_sets, get_outputs_loss_mtl, get_preds_mtl,
        device, log=True, options=options, ema=ema, start_epoch=start_epoch, on_epoch_end=on_epoch_end,
        stepper_state=stepper_state)
    torch.save(_final_weights(net, ema), checkpoint_path)
    logger.info(f"Saved checkpoint {checkpoint_path}" + (" (EMA weights)" if ema is not None else ""))


def train_stage(stage: PedRecTrainingStage, description: ExperimentDescription, net, device: torch.device,
                cycle_num: int, epochs_round_1: int, epochs_round_2: int, lr: float = None,
                force_lr_finder: bool = False, skip_round_1: bool = False, options: TrainingOptions = None,
                resume: bool = False):
    options = options or TrainingOptions()
    paths = description.experiment_paths
    os.makedirs(paths.output_dir, exist_ok=True)
    stage_state = StageState(os.path.join(paths.output_dir, f"{stage.experiment_name}_{cycle_num}_state.pth"))
    if resume and not stage_state.load():
        logger.warning(f"--resume: no state file {stage_state.path}, starting from scratch")
    if stage_state.loaded is None and cycle_num > 0:
        net.load_state_dict(load_state_dict_file(paths.get_stage_checkpoint_path(stage.name, cycle_num - 1)))

    split_params = split_no_wd_params(description.net_layers)
    params = [{'params': p, 'weight_decay': 0 if wd else 1e-2} for (wd, p) in split_params]

    train_loader = get_train_loader(description, IMAGENET_TRANSFORM)
    validation_sets = get_validation_sets(description, IMAGENET_TRANSFORM)

    if stage_state.loaded is not None:
        description.suggested_lr = stage_state.loaded["suggested_lr"]
    elif lr is not None:
        description.suggested_lr = lr
    elif stage.fixed_lr is not None and not force_lr_finder:
        description.suggested_lr = stage.fixed_lr
    else:
        plot_path = os.path.join(paths.output_dir, f"{stage.experiment_name}_lr_range_test.png")
        description.suggested_lr = find_learning_rate(net, params, train_loader, device, plot_path)
    logger.info(f"Used LR: {description.suggested_lr:.2e} | {options.describe()} | "
                f"effective batch size {description.batch_size * options.accumulate}")

    # Append the MTL sigmas AFTER the LR range test.
    params.append({'params': net.loss_head.sigmas, 'weight_decay': 1e-2})

    ema = TrainStepper.create_ema(net, options)
    if stage_state.loaded is not None:
        net.load_state_dict(stage_state.loaded["model"])
        if ema is not None and stage_state.loaded.get("ema") is not None:
            ema.load_state_dict(stage_state.loaded["ema"])
        rng = stage_state.loaded["rng"]
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
    resume_round = stage_state.loaded["round"] if stage_state.loaded is not None else 0
    best_checkpoint = os.path.join(paths.output_dir, f"{stage.experiment_name}_{cycle_num}_best.pth")

    # Round 1: frozen feature extractor, full lr on the heads
    round_1_checkpoint = paths.get_stage_checkpoint_path(stage.name, cycle_num, round_suffix="01")
    if skip_round_1 and stage_state.loaded is None:
        logger.info(f"Skipping round 1, loading {round_1_checkpoint}")
        net.load_state_dict(load_state_dict_file(round_1_checkpoint))
        if ema is not None:
            ema = TrainStepper.create_ema(net, options)
    elif resume_round == 0:
        run_round(0, net, description, params,
                  max_lrs=get_max_lrs(stage, description.suggested_lr),
                  epochs=epochs_round_1,
                  frozen_layers=[net.model.feature_extractor],
                  train_loader=train_loader, validation_sets=validation_sets, device=device,
                  checkpoint_path=round_1_checkpoint, options=options, ema=ema, stage_state=stage_state,
                  best_checkpoint_path=best_checkpoint)
        stage_state.loaded = None

    # Round 2: everything trainable with reduced lr
    run_round(1, net, description, params,
              max_lrs=get_max_lrs(stage, description.suggested_lr, feature_extractor_divisor=20, head_divisor=10),
              epochs=epochs_round_2,
              frozen_layers=[],
              train_loader=train_loader, validation_sets=validation_sets, device=device,
              checkpoint_path=paths.get_stage_checkpoint_path(stage.name, cycle_num), options=options, ema=ema,
              stage_state=stage_state, best_checkpoint_path=best_checkpoint)

    if stage_state.best.best_epoch is not None:
        logger.info(f"Best epoch: {stage_state.best.best_epoch} -> {best_checkpoint}")
    protocol = get_experiment_protocol(description)
    save_log(protocol, paths.get_stage_protocol_path(stage.name))
    print(protocol)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default=DEFAULT_STAGE, choices=sorted(STAGES.keys()), metavar="STAGE",
                        help=f"Training stage to run (default: {DEFAULT_STAGE}). See --list.")
    parser.add_argument("--list", action="store_true", help="List all training stages and exit.")
    parser.add_argument("--chain", action="store_true",
                        help="Print the stages which need to be trained before --stage and exit.")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--output-dir", default=None,
                        help="Checkpoint / protocol output directory (default: <data-dir>/models/pedrec/single_results).")
    parser.add_argument("--init-weights", default=None,
                        help="Explicit checkpoint to initialize from instead of the predecessor stage checkpoint.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: stage default, 48).")
    parser.add_argument("--num-workers", type=int, default=12, help="DataLoader workers (default: 12).")
    parser.add_argument("--epochs-round-1", type=int, default=10, help="Epochs with frozen backbone (default: 10).")
    parser.add_argument("--epochs-round-2", type=int, default=5, help="Epochs with full network (default: 5).")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate, skips the LR range test.")
    parser.add_argument("--lr-finder", action="store_true",
                        help="Always run the LR range test, even if the stage defines a fixed lr.")
    parser.add_argument("--cycle", type=int, default=0,
                        help="Training cycle; cycle > 0 continues from the checkpoint of cycle - 1 (default: 0).")
    parser.add_argument("--skip-round-1", action="store_true",
                        help="Load the round 1 checkpoint (*_01.pth) instead of training round 1.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted training from <output-dir>/experiment_pedrec_<stage>_<cycle>_state.pth.")
    stability = parser.add_argument_group("numerics / stability")
    stability.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto",
                           help="Mixed precision (default auto: bf16 on GPUs that support it, e.g. RTX 30xx-50xx). "
                                "'off' reproduces the original fp32 training.")
    stability.add_argument("--grad-clip", type=float, default=10.0, help="Max gradient norm, 0 disables (default 10).")
    stability.add_argument("--accumulate", type=int, default=1,
                           help="Gradient accumulation steps, e.g. --batch-size 24 --accumulate 2 on 12 GB GPUs.")
    stability.add_argument("--ema-decay", type=float, default=0.9998,
                           help="EMA of the weights used for validation and the checkpoints, 0 disables.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--numexpr-threads", type=int, default=16)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        print(format_stage_table())
        return
    if args.chain:
        print(" -> ".join(stage.name for stage in get_stage_chain(args.stage)))
        return

    os.environ['NUMEXPR_MAX_THREADS'] = str(args.numexpr_threads)
    stage = get_stage(args.stage)
    experiment_paths = get_experiment_paths(args.data_dir)
    if args.output_dir is not None:
        experiment_paths.output_dir = args.output_dir
    description = get_experiment_description(stage, experiment_paths,
                                             batch_size=args.batch_size or stage.batch_size,
                                             num_workers=args.num_workers)
    init_experiment(description.seed)
    device = get_device(use_gpu=not args.cpu)
    logger.info(f"Training stage '{stage.name}': {stage.description}")
    net = build_net(stage, description, device, init_weights_path=args.init_weights)
    train_stage(stage, description, net, device,
                cycle_num=args.cycle,
                epochs_round_1=args.epochs_round_1,
                epochs_round_2=args.epochs_round_2,
                lr=args.lr,
                force_lr_finder=args.lr_finder,
                skip_round_1=args.skip_round_1,
                options=TrainingOptions(amp=args.amp, grad_clip=args.grad_clip, accumulate=args.accumulate,
                                        ema_decay=args.ema_decay),
                resume=args.resume)


if __name__ == '__main__':
    main()
