"""
PedRec demo / inference entry point.

Runs the PedRec pipeline (RT-DETR detection -> PedRecNet pose / orientation -> ByteTrack -> ST-GCN action
recognition) on a video, an image directory, a single image or a webcam, either in the Qt GUI or headless (writing an annotated
video / images and optionally a JSON file with all results).

Examples:
    python pedrec/demo.py --video data/demo/multi_person_crossing_street.mp4
    python pedrec/demo.py --webcam 0 --size 1280x720
    python pedrec/demo.py --images path/to/frames --headless --output out.mp4 --json out.json
    python pedrec/demo.py --image person.jpg --no-detector --no-tracking --no-action --headless --output out.jpg
    python pedrec/demo.py --video in.mp4 --no-pose --headless --output detections.mp4   # detector only

PedRecNet / ST-GCN weights are looked up below the data root (``--data-dir`` / ``PEDREC_DATA_DIR``, default ``data``),
the RT-DETR detector is loaded from the Hugging Face cache (``mise run download:rtdetr``), see README.
"""
import argparse
import json
import logging
import os
import sys

sys.path.append('.')

import cv2
import numpy as np

from pedrec.configs.app_config import AppConfig, action_list_c01, action_list_c01_w_real
from pedrec.configs import default_paths
from pedrec.inference.pipeline import PedRecPipeline, PipelineConfig, FrameResult, RuntimeConfig
from pedrec.models.data_structures import ImageSize
from pedrec.models.human import Human
from pedrec.utils.file_helper import get_img_paths_from_folder
from pedrec.utils.input_providers.img_dir_provider import ImgDirProvider
from pedrec.utils.input_providers.img_provider import ImgProvider
from pedrec.utils.input_providers.input_provider_base import InputProviderBase
from pedrec.utils.input_providers.prefetch_provider import PrefetchProvider
from pedrec.utils.input_providers.video_provider import VideoProvider
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.torch_utils.torch_helper import get_device
from pedrec.visualizers.frame_visualizer import draw_frame_result

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov", ".m4v")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_size(value: str) -> ImageSize:
    try:
        width, height = value.lower().split("x")
        return ImageSize(width=int(width), height=int(height))
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid size '{value}', expected WIDTHxHEIGHT (e.g. 1920x1080)")


def probe_source_size(args) -> ImageSize:
    """Determines the frame size of the input if --size was not given."""
    if args.size is not None:
        return args.size
    if args.webcam is not None:
        return ImageSize(width=1280, height=720)
    if args.video is not None:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video {args.video}")
        size = ImageSize(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        return size
    img_path = args.image
    if args.images is not None:
        img_paths = sorted(get_img_paths_from_folder(args.images))
        if len(img_paths) == 0:
            raise FileNotFoundError(f"No images found in {args.images}")
        img_path = img_paths[0]
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image {img_path}")
    return ImageSize(width=img.shape[1], height=img.shape[0])


def probe_source_fps(args) -> float:
    """Frame rate of the input (used for smoothing and the temporal sampling of the action recognition)."""
    if args.fps is not None:
        return float(args.fps)
    if args.video is not None or args.webcam is not None:
        cap = cv2.VideoCapture(args.video if args.video is not None else args.webcam)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
        cap.release()
        if fps and 1 <= fps <= 240:
            return float(fps)
    return 30.0


def get_input_provider(args, img_size: ImageSize) -> InputProviderBase:
    if args.webcam is not None:
        provider = VideoProvider(args.webcam, img_size, mirror=args.mirror)
    elif args.video is not None:
        provider = VideoProvider(args.video, img_size, mirror=args.mirror)
    elif args.images is not None:
        provider = ImgDirProvider(args.images, fps=args.fps, image_size=img_size)
    else:
        return ImgProvider(args.image, img_size)
    if args.prefetch > 0:
        provider = PrefetchProvider(provider, queue_size=args.prefetch)
    return provider


def get_pipeline_config(args) -> PipelineConfig:
    return PipelineConfig(
        use_detector=not args.no_detector,
        use_pose=not args.no_pose,
        use_tracking=not args.no_tracking,
        use_action=not args.no_action,
        pedrec_weights=args.pedrec_weights,
        ehpi3d_weights=args.ehpi3d_weights,
        data_root=args.data_dir,
        detector_conf_thresh=args.detector_conf_thresh,
        human_min_score=args.human_min_score,
        action_thresh=args.action_thresh,
        rtdetr_model=args.rtdetr_model,
        source_fps=args.source_fps,
        runtime=RuntimeConfig(half=args.half, channels_last=args.channels_last, compile=args.compile),
    )


def human_to_dict(human: Human) -> dict:
    return {
        "uid": int(human.uid),
        "score": float(human.score),
        "bb": [float(v) for v in human.bb],
        "skeleton_2d": human.skeleton_2d.tolist() if human.skeleton_2d is not None else None,
        "skeleton_3d": human.skeleton_3d.tolist() if human.skeleton_3d is not None else None,
        "orientation": human.orientation.tolist() if human.orientation is not None else None,
        "actions": [action.name for action in human.actions] if human.actions else [],
        "action_probabilities": human.action_probabilities.tolist() if human.action_probabilities is not None else None,
    }


def result_to_dict(result: FrameResult) -> dict:
    return {
        "frame_nr": result.frame_nr,
        "fps": result.fps,
        "timings": {k: round(v, 4) for k, v in result.timings.items()},
        "humans": [human_to_dict(human) for human in result.humans],
        "objects": [[float(v) for v in bb] for bb in result.objects],
    }


class OutputWriter:
    """Writes annotated frames to a video file or (as images) into a directory."""

    def __init__(self, output: str, img_size: ImageSize, fps: int):
        self.output = output
        self.img_size = img_size
        self.fps = fps
        self.writer = None
        self.is_video = output.lower().endswith(VIDEO_EXTENSIONS)
        self.is_image = output.lower().endswith(IMAGE_EXTENSIONS)
        if self.is_video:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
            self.writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                          (img_size.width, img_size.height))
        elif self.is_image:
            os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        else:
            os.makedirs(output, exist_ok=True)

    def write(self, frame_nr: int, img_rgb: np.ndarray):
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if self.is_video:
            self.writer.write(img_bgr)
        elif self.is_image:
            cv2.imwrite(self.output, img_bgr)
        else:
            cv2.imwrite(os.path.join(self.output, f"frame_{frame_nr:06d}.jpg"), img_bgr)

    def close(self):
        if self.writer is not None:
            self.writer.release()


def run_headless(pipeline: PedRecPipeline, input_provider: InputProviderBase, img_size: ImageSize, args):
    writer = OutputWriter(args.output, img_size, args.fps or int(round(args.source_fps))) if args.output else None
    results = []
    frame_nr = 0
    try:
        for img in input_provider.get_data():
            frame_nr += 1
            result = pipeline.process(frame_nr, img)
            if args.json:
                results.append(result_to_dict(result))
            if writer is not None:
                writer.write(frame_nr, draw_frame_result(img.copy(), result.humans, result.objects, result.fps))
            if frame_nr % args.log_every == 0 or args.image is not None:
                actions = {human.uid: [a.name for a in human.actions] for human in result.humans if human.actions}
                logger.info(f"frame {frame_nr}: {len(result.humans)} humans, {len(result.objects)} objects, "
                            f"{result.fps} fps, actions: {actions}")
            if args.max_frames and frame_nr >= args.max_frames:
                break
    finally:
        if hasattr(input_provider, "stop"):
            input_provider.stop()
        if writer is not None:
            writer.close()
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(results, f)
        logger.info(f"Wrote {len(results)} frame results to {args.json}")
    logger.info(f"Processed {frame_nr} frames")


def run_gui(pipeline: PedRecPipeline, input_provider: InputProviderBase, app_cfg: AppConfig):
    os.environ.setdefault("QT_API", "pyqt6")
    from qtpy.QtWidgets import QApplication
    from pedrec.ui.pedrec_app import PedRecApp
    from pedrec.ui.pipeline_worker import PipelineWorker

    app = QApplication(sys.argv)
    worker = PipelineWorker(app, input_provider, pipeline)
    PedRecApp(app, worker, app_cfg)
    sys.exit(app.exec())


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", help="Video file (default: <data-dir>/demo/multi_person_crossing_street.mp4).")
    source.add_argument("--images", help="Directory with an image sequence.")
    source.add_argument("--image", help="Single image.")
    source.add_argument("--webcam", type=int, nargs="?", const=0, metavar="ID", help="Webcam id (default 0).")
    parser.add_argument("--size", type=parse_size, default=None,
                        help="Processing size WIDTHxHEIGHT (default: source size, webcam 1280x720).")
    parser.add_argument("--fps", type=int, default=None, help="Playback fps for image directories / output video.")
    parser.add_argument("--mirror", action="store_true", help="Mirror video / webcam frames.")

    components = parser.add_argument_group("components")
    components.add_argument("--no-detector", action="store_true",
                            help="Skip the detector, use the full frame as the human bounding box.")
    components.add_argument("--no-pose", action="store_true", help="Skip PedRecNet (detector only).")
    components.add_argument("--no-tracking", action="store_true", help="Skip tracking / id assignment / smoothing.")
    components.add_argument("--no-action", action="store_true", help="Skip the action recognition.")
    components.add_argument("--action-list", choices=["c01", "c01_real"], default="c01_real",
                            help="Action classes of the action recognition weights (default: c01_real, 20 classes).")

    runtime = parser.add_argument_group("runtime / speed")
    runtime.add_argument("--half", action="store_true", help="fp16 autocast on CUDA (Tensor Cores).")
    runtime.add_argument("--channels-last", action="store_true", help="NHWC memory format for the convolutions.")
    runtime.add_argument("--compile", action="store_true", help="torch.compile PedRecNet / ST-GCN (slow warm up).")
    runtime.add_argument("--fast", action="store_true", help="Shortcut for --half --channels-last on CUDA.")
    runtime.add_argument("--prefetch", type=int, default=4,
                         help="Frames decoded ahead in a background thread (0 disables, default 4).")

    weights = parser.add_argument_group("weights")
    weights.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    weights.add_argument("--rtdetr-model", default=default_paths.RTDETR_MODEL,
                         help=f"RT-DETR Hugging Face id or local directory (default {default_paths.RTDETR_MODEL}).")
    weights.add_argument("--pedrec-weights", default=None,
                         help=f"Default: <data-dir>/{default_paths.PEDREC_NET_WEIGHTS}")
    weights.add_argument("--ehpi3d-weights", default=None, help=f"Default: <data-dir>/{default_paths.EHPI3D_WEIGHTS}")

    thresholds = parser.add_argument_group("thresholds")
    thresholds.add_argument("--detector-conf-thresh", type=float, default=0.4)
    thresholds.add_argument("--human-min-score", type=float, default=0.65)
    thresholds.add_argument("--action-thresh", type=float, default=0.7)

    output = parser.add_argument_group("headless output")
    output.add_argument("--headless", action="store_true", help="Run without the Qt GUI.")
    output.add_argument("--output", default=None,
                        help="Annotated output: video file (.mp4/.avi), image file or directory for frames.")
    output.add_argument("--json", default=None, help="Write all per-frame results (humans, bbs, poses) as JSON.")
    output.add_argument("--max-frames", type=int, default=None, help="Stop after N frames.")
    output.add_argument("--log-every", type=int, default=30, help="Log a summary every N frames (default 30).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference.")
    args = parser.parse_args(argv)
    if args.video is None and args.images is None and args.image is None and args.webcam is None:
        args.video = default_paths.demo_video(args.data_dir)
    if args.no_pose:
        args.no_tracking = True
        args.no_action = True
    if args.no_tracking:
        args.no_action = True
    if args.fast:
        args.half = True
        args.channels_last = True
    return args


def main(argv=None):
    configure_logger()
    args = parse_args(argv)

    app_cfg = AppConfig()
    app_cfg.inference.action_list = action_list_c01 if args.action_list == "c01" else action_list_c01_w_real
    app_cfg.inference.img_size = probe_source_size(args)
    app_cfg.cuda.use_gpu = not args.cpu
    args.source_fps = probe_source_fps(args)
    logger.info(f"Processing size: {app_cfg.inference.img_size.width}x{app_cfg.inference.img_size.height}, "
                f"source fps: {args.source_fps:.1f}")

    device = get_device(app_cfg.cuda.use_gpu)
    pipeline = PedRecPipeline(get_pipeline_config(args), app_cfg, device)
    input_provider = get_input_provider(args, app_cfg.inference.img_size)

    if args.headless:
        run_headless(pipeline, input_provider, app_cfg.inference.img_size, args)
    else:
        run_gui(pipeline, input_provider, app_cfg)


if __name__ == '__main__':
    main()
