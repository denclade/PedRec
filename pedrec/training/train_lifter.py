"""Train a causal lifter on real canonical predictions, noisy GT and clean GT.

Run mise run train:lifter:data first. Prediction inputs remain canonical; their
loss is evaluated in the GT convention with the frozen matching PoseNet adapter.
GT/noise pairs retain their native convention and use direct denoising/identity
losses. Validation labels real-prediction and synthetic-noise inputs explicitly.
"""
import argparse
import json
import logging
import os
import sys
from typing import Dict

sys.path.append(".")

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR

from pedrec.configs import default_paths
from pedrec.datasets.lifter_dataset_specs import LifterDatasetSpec, dataset_specs, dataframe_signature, file_sha256
from pedrec.datasets.pose_sequence_dataset import (PoseSequenceDataset, LifterTrainingMixture,
                                                   PREDICTION, NOISY_GT, SOURCE_NAMES)
from pedrec.networks.net_pedrec.pedrec_convention_adapter import ConventionAdapter
from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file
from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.networks.net_pedrec.pose_lifter import TemporalPoseLifter, SKELETON_3D_RANGE
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.experiment_dataset_helper import WORKER_CONTEXT
from pedrec.training.experiments.experiment_train_helper import init_experiment
from pedrec.training.experiments.train_stepper import TrainingOptions, TrainStepper, steps_per_epoch
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.torch_utils.torch_helper import get_device

logger = logging.getLogger(__name__)


def mpjpe(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked mean per joint position error (in the units of the inputs)."""
    error = torch.linalg.vector_norm(pred - target, dim=-1)
    return (error * mask).sum() / mask.sum().clamp_min(1)


def get_dataset_specs(paths):
    specs = dataset_specs(paths)
    return [s for s in specs if s.split == "train"], [s for s in specs if s.split == "val"]


def _spec_fields(spec):
    if isinstance(spec, LifterDatasetSpec):
        return (spec.title + (" val" if spec.split == "val" else ""), spec.dataframe,
                spec.results if os.path.isfile(spec.results) else None, spec.convention)
    name, dataframe, results = spec  # compatibility with programmatic mini-dataset callers
    return name, dataframe, results, -1


def check_prediction_manifest(spec, weights_hash):
    if not isinstance(spec, LifterDatasetSpec) or not os.path.isfile(spec.results):
        return
    manifest_path = spec.results + ".json"
    if not os.path.isfile(manifest_path):
        logger.warning("%s: predictions have no provenance manifest; regenerate with train:lifter:data", spec.results)
        return
    with open(manifest_path) as stream:
        metadata = json.load(stream)
    if not metadata.get("canonical") or metadata.get("dataframe_signature") != dataframe_signature(spec.dataframe):
        raise ValueError(f"{spec.results}: sequence dataframe changed or predictions are not canonical; regenerate train:lifter:data")
    if weights_hash is not None and metadata.get("pose_weights_sha256") != weights_hash:
        raise ValueError(f"{spec.results}: predictions use different pose weights; regenerate with the selected --pose-weights")


def load_pose_adapter(checkpoint, device):
    state = load_state_dict_file(checkpoint)
    offsets = state.get("loss_head.convention_adapter.offsets")
    if offsets is None:
        raise ValueError("Convention-aware lifter training needs the PoseNet MTL checkpoint (including its adapter), not *_net.pth")
    adapter = ConventionAdapter()
    adapter.offsets.data.copy_(offsets)
    return adapter.to(device).eval().requires_grad_(False)


def adapt_for_loss(pose, convention, adapter):
    if not (convention >= 0).any():
        return pose
    if adapter is None:
        raise ValueError("Canonical predictions with non-SIM GT require the matching frozen pose adapter")
    dummy_2d = torch.zeros((*pose.shape[:2], 2), dtype=pose.dtype, device=pose.device)
    return adapter(dummy_2d, pose, convention)[1]


def _batch_to_device(batch, device):
    features, target, mask = [value.to(device) for value in batch[:3]]
    metadata = batch[3] if len(batch) > 3 else {}
    convention = metadata.get("convention", torch.full((len(features),), -1)).to(device)
    return features, target, mask, convention, metadata


@torch.inference_mode()
def evaluate(model, loader, device, pose_adapter=None) -> Dict[str, float]:
    model.eval()
    sums = {"per_frame": 0.0, "lifted": 0.0, "joints": 0.0}
    for batch in loader:
        features, target, mask, convention, _ = _batch_to_device(batch, device)
        pred = adapt_for_loss(model(features), convention, pose_adapter)
        baseline = adapt_for_loss(features[:, -1, :, 3:6], convention, pose_adapter)
        joints = mask.sum().item()
        sums["per_frame"] += mpjpe(baseline, target, mask).item() * joints
        sums["lifted"] += mpjpe(pred, target, mask).item() * joints
        sums["joints"] += joints
    if not sums["joints"]:
        raise ValueError("Validation has no supported target joints")
    return {"per_frame_mm": sums["per_frame"] / sums["joints"] * SKELETON_3D_RANGE,
            "lifted_mm": sums["lifted"] / sums["joints"] * SKELETON_3D_RANGE}


def train_lifter(train_specs, val_specs, device: torch.device, output_path: str, epochs: int = 30,
                 batch_size: int = 1024, lr: float = 2e-3, num_workers: int = 8, target_step: int = 1,
                 gt_result_ratio: float = 0.5, options: TrainingOptions = None, prediction_ratio=None,
                 clean_gt_ratio: float = 0.1, samples_per_epoch=None, pose_adapter=None, pose_weights_hash=None):
    options = options or TrainingOptions(ema_decay=0.999)
    if not train_specs:
        raise FileNotFoundError("No training dataframes found; see README")
    prediction_ratio = 1 - gt_result_ratio if prediction_ratio is None else prediction_ratio
    if pose_adapter is not None:
        pose_adapter = pose_adapter.to(device).eval().requires_grad_(False)
    for spec in train_specs + val_specs:
        check_prediction_manifest(spec, pose_weights_hash)
    train_datasets = []
    for spec in train_specs:
        name, dataframe, results, convention = _spec_fields(spec)
        dataset = PoseSequenceDataset(dataframe, results, train=True, target_step=target_step,
                                      convention=convention, return_metadata=True)
        if not len(dataset):
            logger.warning("%s: no valid training targets, skipped", name)
            continue
        if convention >= 0 and dataset.valid_result.any() and prediction_ratio > 0 and pose_adapter is None:
            raise ValueError(f"{name}: load the matching pose adapter for canonical prediction inputs")
        train_datasets.append(dataset)
    train_set = LifterTrainingMixture(train_datasets, prediction_ratio, clean_gt_ratio, samples_per_epoch)
    logger.info("Training source probabilities: predictions %.0f%%, noisy GT %.0f%%, clean GT %.0f%%; "
                "uniform dataset sampling within each source; %d samples/epoch",
                100 * prediction_ratio, 100 * (1 - prediction_ratio - clean_gt_ratio), 100 * clean_gt_ratio, len(train_set))
    context = WORKER_CONTEXT if num_workers > 0 else None
    val_loaders, real_val_names = {}, []
    for spec in val_specs:
        name, dataframe, results, convention = _spec_fields(spec)
        # Real-prediction validation never substitutes noisy GT for missing images.
        dataset = PoseSequenceDataset(dataframe, results, train=False, target_step=max(1, target_step),
                                      convention=convention, return_metadata=True,
                                      source=PREDICTION if results is not None else NOISY_GT)
        if not len(dataset):
            logger.warning("%s: no valid validation targets for %s inputs, skipped", name,
                           "prediction" if results is not None else "synthetic-noise")
            continue
        logger.info("%s: validation inputs=%s, %d targets", name,
                    "real predictions (loss in GT convention)" if results is not None else "synthetic GT noise", len(dataset))
        val_loaders[name] = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                                       multiprocessing_context=context)
        if results is not None:
            real_val_names.append(name)
    if not val_loaders:
        raise ValueError("No usable validation targets; cannot select a lifter checkpoint")
    score_sets = real_val_names or list(val_loaders)
    if not real_val_names:
        logger.warning("No real-prediction validation set; checkpoint selection uses synthetic noise only")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              persistent_workers=num_workers > 0, multiprocessing_context=context)
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
        source_counts = torch.zeros(3, dtype=torch.long)
        for i, batch in enumerate(train_loader):
            features, target, mask, convention, metadata = _batch_to_device(batch, device)
            source_counts += torch.bincount(metadata["source"], minlength=3)
            with stepper.autocast():
                pred = model(features)
            loss = mpjpe(adapt_for_loss(pred.float(), convention, pose_adapter), target, mask)
            stepper.step(loss, last_batch=i == len(train_loader) - 1)
            total += loss.item()
            batches += 1
        eval_model = ema.module if ema is not None else model
        results = {name: evaluate(eval_model, loader, device, pose_adapter) for name, loader in val_loaders.items()}
        summary = " | ".join(f"{name} [{'pred' if name in real_val_names else 'GT+noise'}]: "
                             f"{r['per_frame_mm']:.1f} -> {r['lifted_mm']:.1f} mm" for name, r in results.items())
        logger.info("epoch %d/%d train MPJPE %.1f mm | val (input -> lifted, native GT convention) %s",
                    epoch + 1, epochs, total / max(batches, 1) * SKELETON_3D_RANGE, summary)
        logger.info("Epoch input counts: %s", dict(zip(SOURCE_NAMES, source_counts.tolist())))
        # Equal dataset weight; relative to each dataset's own baseline. Empty
        # validation sets never contribute fake 0 mm to this score.
        score = sum(results[name]["lifted_mm"] / max(results[name]["per_frame_mm"], 1e-6)
                    for name in score_sets) / len(score_sets)
        if score < best:
            best, best_results = score, results
            torch.save(eval_model.state_dict(), output_path)
            with open(output_path + ".json", "w") as stream:
                json.dump({"epoch": epoch + 1, "pose_weights_sha256": pose_weights_hash,
                           "prediction_ratio": prediction_ratio, "clean_gt_ratio": clean_gt_ratio,
                           "input_counts": dict(zip(SOURCE_NAMES, source_counts.tolist())),
                           "frozen_adapter_for_prediction_loss": pose_adapter is not None,
                           "dataset_sampling": "uniform within input source", "selection_sets": score_sets,
                           "relative_score": score, "validation": results}, stream, indent=2)
            logger.info("Saved %s (relative validation score %.4f)", output_path, score)
    for name, result in best_results.items():
        if result["lifted_mm"] >= result["per_frame_mm"]:
            logger.warning("%s: lifted poses are not better than inputs (%.1f vs %.1f mm)",
                           name, result["lifted_mm"], result["per_frame_mm"])
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
    ratio = parser.add_mutually_exclusive_group()
    ratio.add_argument("--prediction-ratio", type=float, default=None,
                       help="Global probability of real prediction inputs (default 0.5)")
    ratio.add_argument("--gt-result-ratio", type=float, default=None,
                       help="Compatibility alias: prediction ratio = 1 minus this value")
    parser.add_argument("--clean-gt-ratio", type=float, default=0.1,
                        help="Global clean GT input probability; remainder uses noisy GT (default 0.1)")
    parser.add_argument("--samples-per-epoch", type=int, default=None,
                        help="Sample budget; default sum of available GT targets, datasets balanced within source")
    parser.add_argument("--pose-weights", default=None,
                        help="Matching PoseNet MTL checkpoint for frozen convention adapter; default final p2d3d_c_o")
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
    device = get_device(not args.cpu)
    pose_weights = args.pose_weights or paths.get_stage_checkpoint_path("p2d3d_c_o")
    prediction_ratio = args.prediction_ratio
    if prediction_ratio is None:
        prediction_ratio = 0.5 if args.gt_result_ratio is None else 1 - args.gt_result_ratio
    if not 0 <= prediction_ratio <= 1 or not 0 <= args.clean_gt_ratio <= 1 - prediction_ratio + 1e-12:
        raise ValueError("Input ratios must be nonnegative and sum to at most 1")
    adapter = load_pose_adapter(pose_weights, device) if any(
        s.convention >= 0 and os.path.isfile(s.results) for s in train_specs + val_specs) else None
    digest = file_sha256(pose_weights) if os.path.isfile(pose_weights) else None
    train_lifter(train_specs, val_specs, device,
                 output_path=args.output or default_paths.lifter_weights(args.data_dir),
                 epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers,
                 target_step=args.target_step, prediction_ratio=prediction_ratio, clean_gt_ratio=args.clean_gt_ratio,
                 samples_per_epoch=args.samples_per_epoch, pose_adapter=adapter, pose_weights_hash=digest,
                 options=TrainingOptions(amp=args.amp, ema_decay=args.ema_decay))


if __name__ == "__main__":
    main()
