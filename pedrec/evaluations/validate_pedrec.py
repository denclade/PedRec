"""
Validates a PedRecNet checkpoint on the COCO / SIM / Human3.6m validation sets (2D / 3D pose, joint confidence,
orientation) using the same validation code as the training loop.

    python pedrec/evaluations/validate_pedrec.py --weights data/models/pedrec/single_results/experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0.pth

Accepts both the training checkpoints (MTL wrapper incl. loss head) and the exported ``*_net.pth`` files.
"""
import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse


import torch
import torchvision.transforms as transforms

from pedrec.networks.net_pedrec.pedrec_net_factory import load_pedrec_net, load_arch, pedrec_config, copy_arch
from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file
from pedrec.configs import default_paths
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.evaluations.validate import print_results
from pedrec.models.experiments.experiment_description import ExperimentDescription
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet, PedRecNetLossHead
from pedrec.networks.net_pedrec.pedrec_net_mtl_wrapper import PedRecNetMTLWrapper
from pedrec.training.experiments.experiment_dataset_helper import get_validation_sets
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.experiment_train_helper import init_experiment, get_outputs_loss_mtl, \
    validate_val_sets
from pedrec.utils.torch_utils.torch_helper import get_device


def get_preds_mtl(outputs):
    return {
        "skeleton": outputs[0].cpu().detach().numpy(),
        "skeleton_3d": outputs[1].cpu().detach().numpy(),
        "orientation": outputs[2].cpu().detach().numpy(),
    }


def load_net(weights_path: str, device: torch.device) -> PedRecNetMTLWrapper:
    """Training checkpoint (MTL wrapper) or exported *_net.pth; the architecture comes from the sidecar file."""
    arch = load_arch(weights_path)
    net = PedRecNetMTLWrapper(load_pedrec_net(weights_path, device, arch),
                              PedRecNetLossHead(device, weighting=arch.mtl_weighting,
                                                orientation_head=arch.orientation_head))
    state_dict = load_state_dict_file(weights_path)
    if any(key.startswith("loss_head.") for key in state_dict.keys()):
        net.load_state_dict(state_dict)
    net.to(device)
    return net


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=None,
                        help=f"PedRecNet checkpoint (default: <data-dir>/{default_paths.PEDREC_NET_WEIGHTS}).")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--no-coco", action="store_true", help="Skip the COCO validation set.")
    parser.add_argument("--no-sim", action="store_true", help="Skip the SIM validation set.")
    parser.add_argument("--no-h36m", action="store_true", help="Skip the Human3.6m validation set.")
    parser.add_argument("--sim-val-subsampling", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    experiment_paths = get_experiment_paths(args.data_dir)
    weights = args.weights or default_paths.pedrec_net_weights(args.data_dir)
    experiment_description = ExperimentDescription(
        net_name="PedRecNet",
        experiment_name="validate_pedrec",
        initialization_notes=f"Loaded from {weights}",
        experiment_paths=experiment_paths,
        net_cfg=pedrec_config(load_arch(weights)),
        use_val_coco=not args.no_coco,
        use_val_sim=not args.no_sim,
        use_val_h36m=not args.no_h36m,
        validate_orientation_sim=True,
        validate_orientation_coco=True,
        validate_joint_conf_sim=True,
        validate_joint_conf_coco=True,
        validate_joint_conf_h36m=True,
        sim_val_subsampling=args.sim_val_subsampling,
        batch_size=args.batch_size,
        batch_size_validate=args.batch_size,
        num_workers=args.num_workers,
    )
    experiment_description.coco_val_dataset_cfg.use_mebow_orientation = True
    for dataset_cfg in (experiment_description.coco_val_dataset_cfg, experiment_description.sim_val_dataset_cfg,
                        experiment_description.h36m_val_dataset_cfg):
        dataset_cfg.udp = experiment_description.net_cfg.arch.udp
    init_experiment(experiment_description.seed)
    device = get_device(use_gpu=not args.cpu)
    net = load_net(weights, device)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    validation_sets = get_validation_sets(experiment_description, trans)
    validation_results = validate_val_sets(net, validation_sets, get_outputs_loss_mtl, get_preds_mtl, device,
                                           experiment_description.net_cfg.model.input_size,
                                           udp=experiment_description.net_cfg.arch.udp)
    for name, results in validation_results.items():
        print(f"\n### {name}")
        print_results(results)


if __name__ == '__main__':
    main()
