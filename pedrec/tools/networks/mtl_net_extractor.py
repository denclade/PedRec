"""
Extracts the plain PedRecNet weights (``*_net.pth``, used by the demo) from a training checkpoint which contains the
MTL wrapper (network + loss head).

    python pedrec/tools/networks/mtl_net_extractor.py --stage p2d3d_c_o_h36m_sim_mebow
    python pedrec/tools/networks/mtl_net_extractor.py --input some_checkpoint.pth --output some_checkpoint_net.pth
"""
import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import os

import torch

from pedrec.networks.net_pedrec.pedrec_net_factory import load_pedrec_net, load_arch, pedrec_config, copy_arch
from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet, PedRecNetLossHead
from pedrec.networks.net_pedrec.pedrec_net_mtl_wrapper import PedRecNetMTLWrapper
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.utils.torch_utils.torch_helper import get_device


def mtl_to_net(mtl_wrapper: torch.nn.Module, mtl_path: str, output_path: str):
    mtl_wrapper.load_state_dict(load_state_dict_file(mtl_path))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(mtl_wrapper.model.state_dict(), output_path)
    print(f"Wrote {output_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default=None,
                        help="Training stage name; reads <output-dir>/experiment_pedrec_<stage>_<cycle>.pth and "
                             "writes <data-dir>/models/pedrec/experiment_pedrec_<stage>_<cycle>_net.pth.")
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--input", default=None, help="Explicit MTL checkpoint path.")
    parser.add_argument("--output", default=None, help="Explicit output path.")
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    args = parser.parse_args(argv)
    if args.input is None and args.stage is None:
        parser.error("either --stage or --input is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    paths = get_experiment_paths(args.data_dir)
    mtl_weights = args.input or paths.get_stage_checkpoint_path(args.stage, args.cycle)
    if args.output is not None:
        output_path = args.output
    elif args.stage is not None:
        output_path = os.path.join(os.path.dirname(paths.output_dir),
                                   os.path.basename(mtl_weights).replace(".pth", "_net.pth"))
    else:
        output_path = os.path.splitext(mtl_weights)[0] + "_net.pth"
    arch = load_arch(mtl_weights)
    net = PedRecNetMTLWrapper(PedRecNet(pedrec_config(arch)),
                              PedRecNetLossHead(get_device(False), weighting=arch.mtl_weighting,
                                                orientation_head=arch.orientation_head))
    mtl_to_net(net, mtl_weights, output_path)
    copy_arch(mtl_weights, output_path)


if __name__ == "__main__":
    main()
