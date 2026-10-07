import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse


import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from pedrec.configs.dataset_configs import get_h36m_val_dataset_cfg_default
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.datasets.pedrec_dataset import PedRecDataset
from pedrec.models.constants.dataset_constants import DatasetType
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.networks.net_pedrec.pedrec_net_mtl_wrapper import PedRecNetMTLWrapper
from pedrec.configs import default_paths
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.utils.torch_utils.torch_helper import get_device

h36m_actions = [
    r"Walking(?: .)*\.",
    r"Greeting(?: .)*\.",
    r"Discussion(?: .)*\.",
    r"Phoning(?: .)*\.",
    r"Waiting(?: .)*\.",
    r"Directions(?: .)*\.",
    r"Posing(?: .)*\.",
    r"Purchases(?: .)*\.",
    r"SittingDown(?: .)*\.",
    r"Smoking(?: .)*\.",
    r"Eating(?: .)*\.",
    r"Sitting(?: .)*\.",
    r"Photo(?: .)*\.",
    r"WalkTogether(?: .)*\.",
    r"WalkDog(?: .)*\.|WalkingDog(?: .)*\."
]


def get_msjpe(output, target):
    mask = target[:, :, 4] == 1
    a = output[mask]
    b = target[mask][:, :3]
    num_visible_joints = b.shape[0]
    if num_visible_joints < 1:
        return 0
    test = torch.norm(a[:, 0:3] - b[:, 0:3], dim=1)
    return torch.mean(test)


def validate_h36m(net, val_loaders, device, skeleton_3d_range: int):
    net.eval()

    all = []
    # all_single = []
    for action, val_loader in zip(h36m_actions, val_loaders):
        msjes = []
        msjpes_single = []
        with torch.no_grad():
            for i, val_data in enumerate(val_loader):
                # if count > 2:
                #     break
                # count += 1
                inputs, labels = val_data
                inputs = inputs.to(device)

                outputs = net(inputs)

                # results_single = get_msjpe(torch.unsqueeze(inputs[:, -1, :, :], dim=1), labels)
                results = get_msjpe(outputs[1], labels["skeleton_3d"].to(device))
                # msjpes_single.append(results_single.cpu().detach().numpy())
                msjes.append(results.cpu().detach().numpy())

        msej_mean = np.mean(np.array(msjes) * skeleton_3d_range)
        # msej_mean_single = np.mean(np.array(msjpes_single) * 1000)
        all.append(msej_mean)
        # all_single.append(msej_mean_single)
        print(f"{action}: {msej_mean}")
    full = round(np.mean(np.array(all)), 1)
    # full_single = round(np.mean(np.array(all_single)), 1)
    print(f"Val MSJPE: {full}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Per-action MPJPE of a PedRecNet (*_net.pth) on the Human3.6m "
                                                 "validation set.")
    parser.add_argument("--weights", default=None,
                        help=f"PedRecNet *_net.pth (default: <data-dir>/{default_paths.PEDREC_NET_WEIGHTS}).")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    weights = args.weights or default_paths.pedrec_net_weights(args.data_dir)
    # Initialize net
    device = get_device(use_gpu=not args.cpu)
    net_cfg = PedRecNet50Config()
    net = PedRecNet(net_cfg)
    net.init_weights()
    net.load_state_dict(torch.load(weights, map_location=device))
    net.to(device)

    # Load H36M validation set
    experiment_paths = get_experiment_paths(args.data_dir)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_loaders = []
    val_sets_length = 0
    val_cfg = get_h36m_val_dataset_cfg_default()
    val_cfg.subsample = args.subsample
    for action in h36m_actions:
        val_set = PedRecDataset(experiment_paths.h36m_val_dir,
                                experiment_paths.h36m_val_filename,
                                DatasetType.VALIDATE, val_cfg,
                                net_cfg.model.input_size,
                                trans,
                                is_h36m=True,
                                action_filters=action)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        val_loaders.append(val_loader)
        set_length = len(val_set)
        print(f"{action}: {set_length}")
        val_sets_length += set_length
    print(f"Val sets length: {val_sets_length}")
    validate_h36m(net, val_loaders, device, val_cfg.skeleton_3d_range)


if __name__ == "__main__":
    main()
