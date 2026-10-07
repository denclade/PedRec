"""
Exports YoloV4, PedRecNet and EHPI3D to ONNX (for onnxruntime / TensorRT) and verifies the exported models against
PyTorch.

    python pedrec/tools/networks/export_onnx.py                   # all three networks, default weights
    python pedrec/tools/networks/export_onnx.py --models pedrecnet --pedrec-weights my_net.pth

The files are written to ``<data-dir>/models/onnx/{yolov4,pedrecnet,ehpi3d}.onnx`` and used by the pipeline with
``python pedrec/demo.py --backend onnx``. PedRecNet and EHPI3D are exported with a dynamic batch dimension (number of
persons); with the TensorRT execution provider the engines are built on first use and cached next to the models.
"""
import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import logging
import os

import numpy as np
import torch

from pedrec.networks.net_pedrec.pedrec_net_factory import load_pedrec_net, load_arch, pedrec_config, copy_arch
from pedrec.configs import default_paths
from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.inference.onnx_runtime import default_onnx_dir
from pedrec.networks.net_pedrec.ehpi_3d_net import Ehpi3DNet
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.utils.demo_helper import get_detector, init_pose_model
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.torch_utils.checkpoint_io import load_state_dict_file

logger = logging.getLogger(__name__)


class PedRecNetInference(torch.nn.Module):
    """Only the outputs needed at inference time (2D pose, 3D pose, orientations), without the heatmaps."""

    def __init__(self, net: PedRecNet):
        super().__init__()
        self.net = net

    def forward(self, x):
        outputs = self.net(x)
        return outputs[0], outputs[1], outputs[2]


def export(module: torch.nn.Module, example: torch.Tensor, path: str, output_names, dynamic_batch: bool,
           opset: int):
    module = module.eval()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    dynamic_shapes = None
    if dynamic_batch:
        dynamic_shapes = ({0: torch.export.Dim("batch", min=1, max=256)},)
    torch.onnx.export(module, (example,), path, input_names=["input"], output_names=output_names, opset_version=opset,
                      dynamo=True, dynamic_shapes=dynamic_shapes, external_data=False)
    logger.info(f"Exported {path}")


def verify(module: torch.nn.Module, path: str, example: torch.Tensor, atol: float) -> float:
    import onnxruntime as ort
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = module(example)
    expected = expected if isinstance(expected, tuple) else (expected,)
    actual = session.run(None, {"input": example.numpy()})
    max_diff = max(float(np.max(np.abs(e.numpy() - a))) for e, a in zip(expected, actual))
    status = "OK" if max_diff <= atol else "MISMATCH"
    logger.info(f"{os.path.basename(path)}: max abs difference to PyTorch {max_diff:.2e} ({status})")
    if max_diff > atol:
        raise RuntimeError(f"ONNX export of {path} differs from PyTorch by {max_diff}")
    return max_diff


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=["yolov4", "pedrecnet", "ehpi3d"],
                        default=["yolov4", "pedrecnet", "ehpi3d"])
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--output-dir", default=None, help="Default: <data-dir>/models/onnx")
    parser.add_argument("--yolo-weights", default=None)
    parser.add_argument("--pedrec-weights", default=None)
    parser.add_argument("--ehpi3d-weights", default=None)
    parser.add_argument("--ehpi-frames", type=int, default=64, help="Temporal field of the EHPI3D model (32 or 64).")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--no-verify", action="store_true", help="Skip the comparison with PyTorch.")
    return parser.parse_args(argv)


def main(argv=None):
    configure_logger()
    for noisy in ("onnx_ir", "onnxscript", "torch.onnx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args(argv)
    output_dir = args.output_dir or default_onnx_dir(args.data_dir)
    cpu = torch.device("cpu")
    torch.manual_seed(0)

    if "yolov4" in args.models:
        cfg = YoloV4Config()
        weights = args.yolo_weights or default_paths.yolo_v4_weights(args.data_dir)
        net = get_detector(cfg, weights, logger, cpu)
        example = torch.rand(1, 3, cfg.model.input_size.height, cfg.model.input_size.width)
        path = os.path.join(output_dir, "yolov4.onnx")
        export(net, example, path, ["detections"], dynamic_batch=False, opset=args.opset)
        if not args.no_verify:
            verify(net, path, example, atol=1e-3)

    if "pedrecnet" in args.models:
        weights = args.pedrec_weights or default_paths.pedrec_net_weights(args.data_dir)
        pose_net = load_pedrec_net(weights, cpu)
        cfg = pose_net.cfg
        net = PedRecNetInference(pose_net)
        example = torch.randn(4, 3, cfg.model.input_size.height, cfg.model.input_size.width)
        path = os.path.join(output_dir, "pedrecnet.onnx")
        export(net, example, path, ["pose_2d", "pose_3d", "orientation"], dynamic_batch=True, opset=args.opset)
        copy_arch(weights, path)
        if not args.no_verify:
            verify(net, path, example, atol=1e-3)
            verify(net, path, torch.randn(1, 3, cfg.model.input_size.height, cfg.model.input_size.width), atol=1e-3)

    if "ehpi3d" in args.models:
        weights = args.ehpi3d_weights or default_paths.ehpi3d_weights(args.data_dir)
        net = Ehpi3DNet(len(AppConfig().inference.action_list))
        net.load_state_dict(load_state_dict_file(weights))
        example = torch.randn(4, 3, 32, args.ehpi_frames)
        path = os.path.join(output_dir, "ehpi3d.onnx")
        export(net, example, path, ["action_logits"], dynamic_batch=True, opset=args.opset)
        if not args.no_verify:
            verify(net, path, example, atol=1e-3)


if __name__ == "__main__":
    main()
