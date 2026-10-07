"""
Training of the temporal 3D lifter (``pedrec.networks.net_pedrec.pose_lifter``) on the 3D ground truth sequences of
Human3.6m, SIM-ROM and SIM-C01. Needs only the dataframes, no images:

    python pedrec/training/train_lifter.py

Inputs are the PedRecNet v2 predictions where available (SIM-C01 ``*_allframes.pkl``, written by
``mise run train:ehpi3d:data``) and otherwise the ground truth with noise that imitates the per frame errors. The
validation reports the 3D error (MPJPE, mm) of the per frame PedRecNet poses and of the lifted poses for every
validation set; the lifter has to be better than the per frame poses on the PedRecNet predictions (SIM-C01 val).
"""
import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

sys.path.append(".")

import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from pedrec.configs import default_paths
from pedrec.datasets.extra_3d_datasets import EXTRA_3D_DATASETS
from pedrec.datasets.pose_sequence_dataset import PoseSequenceDataset
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.networks.net_pedrec.pose_lifter import TemporalPoseLifter, SKELETON_3D_RANGE
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.experiment_train_helper import init_experiment
from pedrec.training.experiments.train_stepper import TrainingOptions, TrainStepper, steps_per_epoch
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.torch_utils.torch_helper import get_device

logger = logging.getLogger(__name__)


def mpjpe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean per joint position error (in the units of the inputs)."""
    error = torch.linalg.vector_norm(pred - target, dim=-1)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def _existing(dataset_dir: str, filename: str):
    path = os.path.join(dataset_dir, filename)
    return path if os.path.isfile(path) else None


def get_dataset_specs(paths) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """(name, dataframe, result dataframe or None) for training and validation; missing files are skipped."""
    train = [("H36M", _existing(paths.h36m_train_dir, paths.h36m_train_filename), None),
             ("SIM-ROM", _existing(paths.sim_train_dir, paths.sim_train_filename), None),
             ("SIM-C01", _existing(paths.sim_c01_dir, paths.sim_c01_filename),
              _existing(paths.sim_c01_dir, paths.sim_c01_results_filename))]
    val = [("SIM-C01 val", _existing(paths.sim_c01_val_dir, paths.sim_c01_val_filename),
            _existing(paths.sim_c01_val_dir, paths.sim_c01_val_results_filename)),
           ("H36M val", _existing(paths.h36m_val_dir, paths.h36m_val_filename), None),
           ("SIM-Circle val", _existing(paths.sim_val_dir, paths.sim_val_filename), None)]
    for name, df_path, _ in train + val:
        if df_path is None:
            logger.warning(f"{name}: dataframe not found, skipped")
    for extra in EXTRA_3D_DATASETS:  # converted additional datasets (sequences), if present
        for split, specs in (("train", train), ("val", val)):
            path = extra.path(paths.datasets_dir, split, "seq")
            if path is not None:
                specs.append((f"{extra.title}" + (" val" if split == "val" else ""), path, None))
                logger.info(f"Additional 3D dataset {extra.title} ({split}): {path}")
    return [s for s in train if s[1] is not None], [s for s in val if s[1] is not None]


@torch.inference_mode()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    sums = {"per_frame": 0.0, "lifted": 0.0, "joints": 0.0}
    for features, target, mask in loader:
        features, target, mask = features.to(device), target.to(device), mask.to(device)
        pred = model(features)
        joints = mask.sum().item()
        sums["per_frame"] += mpjpe(features[:, -1, :, 3:6], target, mask).item() * joints
        sums["lifted"] += mpjpe(pred, target, mask).item() * joints
        sums["joints"] += joints
    joints = max(sums["joints"], 1)
    return {"per_frame_mm": sums["per_frame"] / joints * SKELETON_3D_RANGE,
            "lifted_mm": sums["lifted"] / joints * SKELETON_3D_RANGE}


def train_lifter(train_specs, val_specs, device: torch.device, output_path: str, epochs: int = 30,
                 batch_size: int = 1024, lr: float = 2e-3, num_workers: int = 8, target_step: int = 1,
                 gt_result_ratio: float = 0.5, options: TrainingOptions = None) -> Dict[str, Dict[str, float]]:
    options = options or TrainingOptions(ema_decay=0.999)
    if not train_specs:
        raise FileNotFoundError("No training dataframe found (Human3.6m / SIM-ROM / SIM-C01), see README.")
    train_set = ConcatDataset([PoseSequenceDataset(df, results, train=True, target_step=target_step,
                                                   gt_result_ratio=gt_result_ratio)
                               for _, df, results in train_specs])
    val_loaders = {name: DataLoader(PoseSequenceDataset(df, results, train=False, target_step=max(1, target_step)),
                                    batch_size=batch_size, num_workers=num_workers)
                   for name, df, results in val_specs}
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
                              persistent_workers=num_workers > 0)

    model = TemporalPoseLifter(len(SKELETON_PEDREC_JOINTS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs,
                           steps_per_epoch=steps_per_epoch(len(train_loader), options.accumulate))
    ema = TrainStepper.create_ema(model, options)
    stepper = TrainStepper(model, optimizer, scheduler, device, options, ema)
    best, best_results = float("inf"), {}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    for epoch in range(epochs):
        model.train()
        total, batches = 0.0, 0
        for i, (features, target, mask) in enumerate(train_loader):
            features, target, mask = features.to(device), target.to(device), mask.to(device)
            with stepper.autocast():
                pred = model(features)
            loss = mpjpe(pred.float(), target, mask)
            stepper.step(loss, last_batch=i == len(train_loader) - 1)
            total += loss.item()
            batches += 1
        eval_model = ema.module if ema is not None else model
        results = {name: evaluate(eval_model, loader, device) for name, loader in val_loaders.items()}
        summary = " | ".join(f"{name}: {r['per_frame_mm']:.1f} -> {r['lifted_mm']:.1f} mm" for name, r in results.items())
        logger.info(f"epoch {epoch + 1}/{epochs} train MPJPE {total / max(batches, 1) * SKELETON_3D_RANGE:.1f} mm | "
                    f"val (per frame -> lifted) {summary}")
        score = sum(r["lifted_mm"] for r in results.values()) / max(len(results), 1) if results else -epoch
        if score < best:
            best, best_results = score, results
            torch.save(eval_model.state_dict(), output_path)
            logger.info(f"Saved {output_path}")
    for name, r in best_results.items():
        if r["lifted_mm"] >= r["per_frame_mm"]:
            logger.warning(f"{name}: the lifted poses are not better than the per frame poses "
                           f"({r['lifted_mm']:.1f} vs {r['per_frame_mm']:.1f} mm)")
    return best_results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--output", default=None, help=f"Default: <data-dir>/{default_paths.LIFTER_WEIGHTS}")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--target-step", type=int, default=1,
                        help="Use every n-th frame as training target (faster epochs on large datasets).")
    parser.add_argument("--gt-result-ratio", type=float, default=0.5,
                        help="Probability to train on noisy ground truth instead of the PedRecNet predictions.")
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    configure_logger()
    args = parse_args(argv)
    init_experiment(42)
    paths = get_experiment_paths(args.data_dir)
    train_specs, val_specs = get_dataset_specs(paths)
    train_lifter(train_specs, val_specs, get_device(not args.cpu),
                 output_path=args.output or default_paths.lifter_weights(args.data_dir),
                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers,
                 target_step=args.target_step, gt_result_ratio=args.gt_result_ratio,
                 options=TrainingOptions(amp=args.amp, ema_decay=args.ema_decay))


if __name__ == "__main__":
    main()
