"""
Benchmarks the inference pipeline: the isolated networks (RT-DETR, PedRecNet on N person crops, ST-GCN on N
sequences) and the full pipeline per stage.

    python pedrec/tools/benchmark_pipeline.py                         # synthetic 1920x1080 frames, 6 persons
    python pedrec/tools/benchmark_pipeline.py --video my.mp4 --fast   # real video, fp16 + channels_last
    python pedrec/tools/benchmark_pipeline.py --random-weights --cpu  # no weight files / downloads needed

Without ``--random-weights`` the default PedRecNet / ST-GCN weights below the data root and the configured RT-DETR
model are used. With ``--random-weights`` the RT-DETR model uses the default transformers RT-DETRv2 configuration.
"""
import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import logging
import statistics
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.inference import gpu_ops
from pedrec.inference.pipeline import PedRecPipeline, PipelineConfig, RuntimeConfig
from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_detr.rtdetr_detector import RTDetrDetector
from pedrec.networks.net_pedrec.ehpi_stgcn import EhpiStGcn
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.torch_utils.torch_helper import get_device

logger = logging.getLogger(__name__)


def synthetic_frames(size: ImageSize, num_persons: int, num_frames: int):
    rng = np.random.default_rng(0)
    background = cv2.GaussianBlur((rng.random((size.height, size.width, 3)) * 255).astype(np.uint8), (9, 9), 3)
    for i in range(num_frames):
        img = background.copy()
        for p in range(num_persons):
            x = int((p + 0.5) * size.width / num_persons + 3 * i)
            cv2.rectangle(img, (x - 40, size.height // 3), (x + 40, size.height - 60), (180, 160, 150), -1)
        yield img


def video_frames(path: str, size: ImageSize, num_frames: int):
    cap = cv2.VideoCapture(path)
    for _ in range(num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(cv2.resize(frame, (size.width, size.height)), cv2.COLOR_BGR2RGB)
    cap.release()


def synthetic_bbs(size: ImageSize, num_persons: int, shift: float = 0.0):
    return [np.array([(p + 0.5) * size.width / num_persons + shift, size.height * 0.6, 100, size.height * 0.6, 0.9, 0],
                     dtype=np.float32) for p in range(num_persons)]


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device, repeats: int, warmup: int = 3):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def random_models(device: torch.device, num_actions: int, half: bool):
    import transformers
    torch.manual_seed(0)
    pose_net = PedRecNet(PedRecNet50Config())
    pose_net.init_weights()
    detr = transformers.RTDetrV2ForObjectDetection(transformers.RTDetrV2Config(num_labels=80))
    detector = RTDetrDetector(device, "random", half=half, model=detr)
    return detector, pose_net, EhpiStGcn(num_actions)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=None, help="Benchmark on a video instead of synthetic frames.")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--persons", type=int, default=6, help="Persons in the synthetic frames / pose batch size.")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=10, help="Repeats of the isolated network timings.")
    parser.add_argument("--random-weights", action="store_true", help="Use randomly initialized networks.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fast", action="store_true", help="--half --channels-last")
    args = parser.parse_args(argv)
    if args.fast:
        args.half = args.channels_last = True
    width, height = args.size.lower().split("x")
    args.img_size = ImageSize(int(width), int(height))
    return args


def main(argv=None):
    configure_logger()
    logging.getLogger("pedrec").setLevel(logging.WARNING)
    args = parse_args(argv)
    device = get_device(not args.cpu)
    app_cfg = AppConfig()
    app_cfg.inference.img_size = args.img_size
    num_actions = len(app_cfg.inference.action_list)

    runtime = RuntimeConfig(half=args.half, channels_last=args.channels_last, compile=args.compile)
    cfg = PipelineConfig(data_root=args.data_dir, human_min_score=0.0, runtime=runtime)
    models = random_models(device, num_actions, args.half) if args.random_weights else (None, None, None)
    pipeline = PedRecPipeline(cfg, app_cfg, device, *models)

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\nDevice: {device_name} | torch {torch.__version__} | size {args.size} | half={args.half} "
          f"channels_last={args.channels_last} compile={args.compile}")

    # ------------------------------------------------------------------ isolated networks
    img = next(synthetic_frames(args.img_size, args.persons, 1))
    bbs = synthetic_bbs(args.img_size, args.persons)
    ehpis = [np.random.default_rng(i).integers(0, 255, (cfg.temporal_field.height, cfg.temporal_field.width, 3),
                                               dtype=np.uint8) for i in range(args.persons)]

    def run_detector():
        with torch.inference_mode():
            pipeline.detector(gpu_ops.frame_to_tensor(img, device), args.img_size, cfg.person_low_thresh)

    def run_pose():
        with torch.inference_mode():
            pipeline.pose_estimator(gpu_ops.frame_to_tensor(img, device), bbs)

    def run_action():
        with torch.inference_mode():
            pipeline.action_recognizer(ehpis)

    print("\nNetworks (median, ms)")
    for name, fn in [("detector (RT-DETR)", run_detector),
                     (f"pose (PedRecNet, {args.persons} persons)", run_pose),
                     (f"action (ST-GCN, {args.persons} persons)", run_action)]:
        print(f"{name:38} {timeit(fn, device, args.repeats) * 1000:8.1f}")

    # ------------------------------------------------------------------ full pipeline
    frames = video_frames(args.video, args.img_size, args.frames) if args.video \
        else synthetic_frames(args.img_size, args.persons, args.frames)
    if args.random_weights and not args.video:
        # a random detector finds nobody: still run it (timing), but continue with the known person bbs
        detect = pipeline.detect
        frame_counter = iter(range(10 ** 9))

        def detect_with_synthetic_persons(frame):
            detect(frame)
            return synthetic_bbs(args.img_size, args.persons, 3 * next(frame_counter)), []
        pipeline.detect = detect_with_synthetic_persons
    timings = defaultdict(list)
    count = 0
    for nr, frame in enumerate(frames, start=1):
        sync(device)
        result = pipeline.process(nr, frame)
        sync(device)
        if nr <= 3:  # warm up (cudnn benchmark, compile)
            continue
        count += 1
        for stage, value in result.timings.items():
            timings[stage].append(value)
    if count == 0:
        print("Not enough frames for the full pipeline benchmark.")
        return
    print(f"\nFull pipeline ({count} frames, median per frame, ms)")
    for stage, values in timings.items():
        print(f"{stage:20} {statistics.median(values) * 1000:8.1f}")
    print(f"{'=> fps':20} {1.0 / statistics.median(timings['total']):8.1f}")


if __name__ == "__main__":
    main()
