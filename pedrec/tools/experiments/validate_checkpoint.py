"""
Runs the training validation (same sets, subsampling and metrics as train_pedrec.py) on a stored checkpoint and logs the
diagnostics of the current code: 2D / 3D bias (median), 3D error per camera axis and per joint, and the canonical
results without the convention adapter (what the inference network outputs).

    python pedrec/tools/experiments/validate_checkpoint.py \\
        data/models/pedrec/single_results/experiment_pedrec_v2_p2d3d_c_o_0_best.pth --stage p2d3d_c_o

Checkpoints of train_pedrec.py (``*_best.pth``, ``*_0.pth``) contain the convention adapter and the MTL weights, so the
values are those of the training validation. ``--per-joint`` limits the per joint table to some sets (default all 3D).
"""
import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import logging
import os
from typing import List, Optional

import numpy as np
import torch

from pedrec.models.constants.skeleton_pedrec import SKELETON_PEDREC_JOINTS
from pedrec.models.experiments.epoch_validation_results import EpochValidationResults
from pedrec.training.experiments.experiment_dataset_helper import get_validation_sets
from pedrec.training.experiments.experiment_log_helper import log_epoch_validation_results
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.experiment_train_helper import init_experiment, validate_val_sets, \
    get_outputs_loss_mtl
from pedrec.training.experiments.pedrec_stages import get_stage, STAGES, DEFAULT_STAGE
from pedrec.training.train_pedrec import IMAGENET_TRANSFORM, build_net, get_experiment_description, get_preds_mtl, \
    log_validation_diagnostics, validate_canonical, log_canonical_results
from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file
from pedrec.utils.torch_utils.torch_helper import get_device

logger = logging.getLogger(__name__)


def format_per_joint_3d(set_name: str, pose3d) -> str:
    """Per joint MPJPE and mean absolute error per camera axis (mm), joints without GT are skipped."""
    lines = [f"{set_name} 3D error per joint (mm):", f"  {'joint':22} {'MPJPE':>7} {'x':>7} {'y':>7} {'depth':>7}"]
    for j, joint in enumerate(SKELETON_PEDREC_JOINTS):
        values = [pose3d.mpjpe[j], pose3d.mean_joint_x_distances[j], pose3d.mean_joint_y_distances[j],
                  pose3d.mean_joint_depth_distances[j]]
        if not np.all(np.isfinite(values)):
            continue
        lines.append(f"  {joint.name:22} " + " ".join(f"{v:7.1f}" for v in values))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", help="state dict of train_pedrec.py (e.g. ..._0_best.pth)")
    parser.add_argument("--stage", default=DEFAULT_STAGE, choices=sorted(STAGES.keys()),
                        help=f"stage of the checkpoint, decides the validated tasks (default {DEFAULT_STAGE})")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: stage default, 64).")
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--per-joint", nargs="*", default=None, metavar="SET",
                        help="sets for the per joint 3D table (default: all sets with 3D, empty = none)")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args(argv)

    stage = get_stage(args.stage)
    description = get_experiment_description(stage, get_experiment_paths(args.data_dir),
                                             batch_size=args.batch_size or stage.batch_size,
                                             num_workers=args.num_workers)
    init_experiment(description.seed)
    device = get_device(use_gpu=not args.cpu)
    net = build_net(stage, description, device, init_weights_path=args.checkpoint)
    # strict: incl. the MTL weights (reset by build_net) and the convention adapter
    net.load_state_dict(load_state_dict_file(args.checkpoint))
    net.to(device)

    validation_sets = get_validation_sets(description, IMAGENET_TRANSFORM)
    with torch.no_grad():
        validation_results = validate_val_sets(net, validation_sets, get_outputs_loss_mtl, get_preds_mtl, device,
                                               description.net_cfg.model.input_size)
    log_epoch_validation_results(EpochValidationResults(epoch=0, train_loss=float("nan"), train_time=0,
                                                        validation_results=validation_results),
                                 title=f"Validation {os.path.basename(args.checkpoint)}")
    log_validation_diagnostics(net, validation_results)
    log_canonical_results(validate_canonical(net, validation_sets, device, description.net_cfg.model.input_size),
                          validation_results)
    for set_name, set_results in validation_results.items():
        if set_results.pose3d is None or (args.per_joint is not None and set_name not in args.per_joint):
            continue
        logger.info(format_per_joint_3d(set_name, set_results.pose3d))


if __name__ == '__main__':
    main()
