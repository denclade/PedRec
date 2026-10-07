"""
Training entry point for the EHPI3D action recognition network (SIM-C01 skeleton sequences).

    python pedrec/training/train_ehpi3d.py --variant gt_pred_64frames

Run ``--list`` to see all variants. Requires the SIM-C01 skeleton dataframes and the PedRecNet result dataframes
(see README, "Datasets", for action recognition) below ``<data-dir>/datasets/Conti01``.
"""
import argparse
import logging
import math
import os
import sys
import time
from typing import Tuple

sys.path.append(".")

import torch
import torch.optim
import torch.utils.data
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file
from pedrec.configs.app_config import AppConfig
from pedrec.configs.dataset_configs import PedRecTemporalDatasetConfig, VideoActionDatasetConfig
from pedrec.datasets.dataset_helper import worker_init_fn
from pedrec.datasets.pedrec_temporal_dataset import PedRecTemporalDataset
from pedrec.datasets.video_action_dataset import VideoActionDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.models.experiments.experiment_paths import ExperimentPaths
from pedrec.networks.net_pedrec.ehpi_3d_net import Ehpi3DNet
from pedrec.training.experiments.ehpi3d_variants import Ehpi3DVariant, get_variant, format_variant_table, VARIANTS, \
    DEFAULT_VARIANT
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.experiment_train_helper import init_experiment
from pedrec.training.experiments.train_stepper import TrainingOptions, TrainStepper, steps_per_epoch
from pedrec.utils.ehpi_helper import ehpi_transform
from pedrec.utils.torch_utils.lr_finder import LRFinder
from pedrec.utils.torch_utils.torch_helper import get_device, split_no_wd_params, move_to_device

logger = logging.getLogger(__name__)

OPTIMIZER_PARAMS = {
    "lr": 1e-3,
    "betas": (0.9, 0.999),
    "eps": 1e-8,
    "weight_decay": 1e-2,
    "amsgrad": False
}


def get_train_loader(experiment_paths: ExperimentPaths, batch_size: int, num_workers: int, action_list,
                     pedrec_cfg: PedRecTemporalDatasetConfig, vid_cfg: VideoActionDatasetConfig = None) -> DataLoader:
    sim_train = PedRecTemporalDataset(experiment_paths.sim_c01_dir,
                                      experiment_paths.sim_c01_filename,
                                      DatasetType.TRAIN,
                                      pedrec_cfg,
                                      action_list,
                                      ehpi_transform,
                                      pose_results_file=experiment_paths.sim_c01_results_filename)
    train_set = sim_train
    if vid_cfg is not None:
        ehpi_vid_dataset = VideoActionDataset(experiment_paths.ehpi_videos_dir,
                                              experiment_paths.ehpi_videos_results_filename,
                                              DatasetType.TRAIN, vid_cfg, action_list, ehpi_transform)
        train_set = ConcatDataset([sim_train, ehpi_vid_dataset])
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      worker_init_fn=worker_init_fn, pin_memory=True, persistent_workers=num_workers > 0)


def find_learning_rate(net, params, criterion, train_loader, device, output_path: str) -> float:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    optimizer = torch.optim.AdamW(params, lr=1e-7, weight_decay=1e-2)
    lr_finder = LRFinder(net, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, end_lr=9e-01, num_iter=100)
    fig, ax = plt.subplots()
    result = lr_finder.plot(ax=ax)
    lr_finder.reset()
    if not isinstance(result, tuple):
        raise RuntimeError("LR range test did not return a suggestion. Pass a learning rate explicitly via --lr.")
    _, suggested_lr = result
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"LR range test suggested lr={suggested_lr:.2e} (plot: {output_path})")
    return suggested_lr


def train_epoch(net: torch.nn.Module, stepper: TrainStepper, criterion, train_loader: DataLoader,
                device: torch.device) -> Tuple[float, float]:
    start = time.time()
    loss_total = 0.0
    finite_batches = 0
    net.train()
    num_batches = len(train_loader)
    with tqdm(total=num_batches) as pbar:
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = move_to_device(labels, device)
            with stepper.autocast():
                outputs = net(inputs)
            with torch.autocast(device_type=device.type, enabled=False):
                loss = criterion(outputs.float(), labels.float())
            loss_value = loss.item()
            stepper.step(loss, last_batch=i == num_batches - 1)
            if math.isfinite(loss_value):
                loss_total += loss_value
                finite_batches += 1
            pbar.set_postfix(loss=f"{loss_value:.4f}")
            pbar.update()
    return loss_total / max(finite_batches, 1), time.time() - start


def train_variant(variant: Ehpi3DVariant, experiment_paths: ExperimentPaths, device: torch.device, epochs: int,
                  batch_size: int, num_workers: int, lr: float = None, init_weights: str = None,
                  options: TrainingOptions = None) -> str:
    options = options or TrainingOptions()
    app_cfg = AppConfig()
    action_list = app_cfg.inference.action_list
    output_dir = experiment_paths.ehpi3d_output_dir
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, f"{variant.experiment_name}.pth")

    net = Ehpi3DNet(len(action_list))
    if init_weights is not None:
        logger.info(f"Initializing from {init_weights}")
        net.load_state_dict(load_state_dict_file(init_weights))
    net.to(device)

    split_params = split_no_wd_params([net])
    params = [{'params': p, 'weight_decay': 0 if wd else 1e-2} for (wd, p) in split_params]

    train_loader = get_train_loader(experiment_paths, batch_size, num_workers, action_list,
                                    variant.get_pedrec_cfg(), variant.get_vid_cfg())
    criterion = BCEWithLogitsLoss()

    if lr is None:
        lr = find_learning_rate(net, params, criterion, train_loader, device,
                                os.path.join(output_dir, f"{variant.experiment_name}_lr_range_test.png"))
    logger.info(f"Used LR: {lr:.2e} | {options.describe()}")

    optimizer = torch.optim.AdamW(params, **OPTIMIZER_PARAMS)
    scheduler = OneCycleLR(optimizer, max_lr=[lr] * len(params), epochs=epochs,
                           steps_per_epoch=steps_per_epoch(len(train_loader), options.accumulate))
    ema = TrainStepper.create_ema(net, options)
    stepper = TrainStepper(net, optimizer, scheduler, device, options, ema=ema)
    for epoch in range(epochs):
        train_loss, train_time = train_epoch(net, stepper, criterion, train_loader, device)
        logger.info(f"Epoch {epoch}: train loss {train_loss:.5f} ({train_time:.0f}s), {stepper.stats.summary()}")

    torch.save((ema.module if ema is not None else net).state_dict(), checkpoint_path)
    logger.info(f"Saved checkpoint {checkpoint_path}" + (" (EMA weights)" if ema is not None else ""))
    return checkpoint_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=sorted(VARIANTS.keys()), metavar="VARIANT",
                        help=f"Training variant (default: {DEFAULT_VARIANT}). See --list.")
    parser.add_argument("--list", action="store_true", help="List all variants and exit.")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--output-dir", default=None,
                        help="Checkpoint output directory (default: <data-dir>/models/ehpi3d).")
    parser.add_argument("--init-weights", default=None, help="Optional EHPI3D checkpoint to start from.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate, skips the LR range test.")
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto",
                        help="Mixed precision (default auto: bf16 where supported); 'off' = original fp32 training.")
    parser.add_argument("--grad-clip", type=float, default=10.0, help="Max gradient norm, 0 disables.")
    parser.add_argument("--accumulate", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="EMA of the weights, 0 disables.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training.")
    parser.add_argument("--numexpr-threads", type=int, default=16)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list:
        print(format_variant_table())
        return
    os.environ['NUMEXPR_MAX_THREADS'] = str(args.numexpr_threads)
    variant = get_variant(args.variant)
    experiment_paths = get_experiment_paths(args.data_dir)
    if args.output_dir is not None:
        experiment_paths.ehpi3d_output_dir = args.output_dir
    init_experiment(42)
    device = get_device(use_gpu=not args.cpu)
    logger.info(f"Training EHPI3D variant '{variant.name}': {variant.description}")
    train_variant(variant, experiment_paths, device, epochs=args.epochs, batch_size=args.batch_size,
                  num_workers=args.num_workers, lr=args.lr, init_weights=args.init_weights,
                  options=TrainingOptions(amp=args.amp, grad_clip=args.grad_clip, accumulate=args.accumulate,
                                          ema_decay=args.ema_decay))


if __name__ == '__main__':
    main()
