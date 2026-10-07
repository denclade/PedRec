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
import sys

sys.path.append(".")

import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import OneCycleLR

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
from pedrec.training.experiments.experiment_train_helper import init_experiment, train_round, get_outputs_loss_mtl
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


def run_round(net, description: ExperimentDescription, params, max_lrs, epochs: int, frozen_layers, train_loader,
              validation_sets, device, checkpoint_path: str):
    optimizer = torch.optim.AdamW(params, **OPTIMIZER_PARAMS)
    scheduler = OneCycleLR(optimizer, max_lr=max_lrs, steps_per_epoch=len(train_loader), epochs=epochs)
    round_description = ExperimentRoundDescription(
        num_epochs=epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        frozen_layers=frozen_layers,
        max_lrs=max_lrs,
        optimizer_parameters=OPTIMIZER_PARAMS
    )
    description.experiment_rounds.append(round_description)
    round_description.validation_results = train_round(net, description, round_description, train_loader,
                                                       validation_sets, get_outputs_loss_mtl, get_preds_mtl, device,
                                                       log=True)
    torch.save(net.state_dict(), checkpoint_path)
    logger.info(f"Saved checkpoint {checkpoint_path}")


def train_stage(stage: PedRecTrainingStage, description: ExperimentDescription, net, device: torch.device,
                cycle_num: int, epochs_round_1: int, epochs_round_2: int, lr: float = None,
                force_lr_finder: bool = False, skip_round_1: bool = False):
    paths = description.experiment_paths
    os.makedirs(paths.output_dir, exist_ok=True)
    if cycle_num > 0:
        net.load_state_dict(torch.load(paths.get_stage_checkpoint_path(stage.name, cycle_num - 1)))

    split_params = split_no_wd_params(description.net_layers)
    params = [{'params': p, 'weight_decay': 0 if wd else 1e-2} for (wd, p) in split_params]

    train_loader = get_train_loader(description, IMAGENET_TRANSFORM)
    validation_sets = get_validation_sets(description, IMAGENET_TRANSFORM)

    if lr is not None:
        description.suggested_lr = lr
    elif stage.fixed_lr is not None and not force_lr_finder:
        description.suggested_lr = stage.fixed_lr
    else:
        plot_path = os.path.join(paths.output_dir, f"{stage.experiment_name}_lr_range_test.png")
        description.suggested_lr = find_learning_rate(net, params, train_loader, device, plot_path)
    logger.info(f"Used LR: {description.suggested_lr:.2e}")

    # Append the MTL sigmas AFTER the LR range test.
    params.append({'params': net.loss_head.sigmas, 'weight_decay': 1e-2})

    # Round 1: frozen feature extractor, full lr on the heads
    round_1_checkpoint = paths.get_stage_checkpoint_path(stage.name, cycle_num, round_suffix="01")
    if skip_round_1:
        logger.info(f"Skipping round 1, loading {round_1_checkpoint}")
        net.load_state_dict(torch.load(round_1_checkpoint))
    else:
        run_round(net, description, params,
                  max_lrs=get_max_lrs(stage, description.suggested_lr),
                  epochs=epochs_round_1,
                  frozen_layers=[net.model.feature_extractor],
                  train_loader=train_loader, validation_sets=validation_sets, device=device,
                  checkpoint_path=round_1_checkpoint)

    # Round 2: everything trainable with reduced lr
    run_round(net, description, params,
              max_lrs=get_max_lrs(stage, description.suggested_lr, feature_extractor_divisor=20, head_divisor=10),
              epochs=epochs_round_2,
              frozen_layers=[],
              train_loader=train_loader, validation_sets=validation_sets, device=device,
              checkpoint_path=paths.get_stage_checkpoint_path(stage.name, cycle_num))

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
                skip_round_1=args.skip_round_1)


if __name__ == '__main__':
    main()
