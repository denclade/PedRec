"""
Benchmarks the inference pipeline per stage and compares the optimized detector / pose path with the original
implementation (cv2 crops per person, numpy post-processing, CPU NMS).

    python pedrec/tools/benchmark_pipeline.py                         # synthetic 1920x1080 frames, 6 persons
    python pedrec/tools/benchmark_pipeline.py --video my.mp4 --fast   # real video, fp16 + channels_last
    python pedrec/tools/benchmark_pipeline.py --random-weights --cpu  # no weight files needed

Without ``--random-weights`` the default weights below the data root are used.
"""
import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import logging
import os
import statistics
import tempfile
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

from pedrec.configs.app_config import AppConfig
from pedrec.configs.pedrec_net_config import PedRecNet50Config
from pedrec.configs.yolo_v4_config import YoloV4Config
from pedrec.inference import gpu_ops
from pedrec.inference.pipeline import PedRecPipeline, PipelineConfig, RuntimeConfig
from pedrec.models.data_structures import ImageSize
from pedrec.networks.net_pedrec.ehpi_3d_net import Ehpi3DNet
from pedrec.networks.net_pedrec.pedrec_net import PedRecNet
from pedrec.networks.net_yolo_v4.yolo_v4_helper import do_detect
from pedrec.networks.net_yolo_v4.yolov4 import YoloV4
from pedrec.utils.log_helper import configure_logger
from pedrec.utils.pose_deconv_helper import pedrec_recognizer
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


def synthetic_bbs(size: ImageSize, num_persons: int):
    return [np.array([(p + 0.5) * size.width / num_persons, size.height * 0.6, 100, size.height * 0.6, 0.9, 0],
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


def write_random_weights(directory: str, num_actions: int):
    torch.manual_seed(0)
    paths = {"yolo": os.path.join(directory, "yolo.pth"), "pedrec": os.path.join(directory, "pedrec.pth"),
             "ehpi": os.path.join(directory, "ehpi.pth")}
    torch.save(YoloV4(YoloV4Config(), inference=True).state_dict(), paths["yolo"])
    net = PedRecNet(PedRecNet50Config())
    net.init_weights()
    torch.save(net.state_dict(), paths["pedrec"])
    torch.save(Ehpi3DNet(num_actions).state_dict(), paths["ehpi"])
    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default=None, help="Benchmark on a video instead of synthetic frames.")
    parser.add_argument("--size", default="1920x1080")
    parser.add_argument("--persons", type=int, default=6, help="Persons in the synthetic frames / pose batch size.")
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=10, help="Repeats of the isolated stage comparison.")
    parser.add_argument("--random-weights", action="store_true", help="Use randomly initialized networks.")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--fast", action="store_true", help="--half --channels-last")
    parser.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    parser.add_argument("--tracker", choices=["bytetrack", "legacy"], default="bytetrack")
    parser.add_argument("--skip-legacy", action="store_true", help="Skip the comparison with the original code.")
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
    weights = {"yolo": None, "pedrec": None, "ehpi": None}
    tmp = None
    if args.random_weights:
        tmp = tempfile.TemporaryDirectory()
        weights = write_random_weights(tmp.name, len(app_cfg.inference.action_list))

    runtime = RuntimeConfig(half=args.half, channels_last=args.channels_last, compile=args.compile,
                            backend=args.backend)
    cfg = PipelineConfig(yolo_weights=weights["yolo"], pedrec_weights=weights["pedrec"],
                         ehpi3d_weights=weights["ehpi"], data_root=args.data_dir, tracker=args.tracker,
                         human_min_score=0.0, runtime=runtime)
    pipeline = PedRecPipeline(cfg, app_cfg, device)

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"\nDevice: {device_name} | torch {torch.__version__} | size {args.size} | half={args.half} "
          f"channels_last={args.channels_last} compile={args.compile} backend={args.backend}")

    # ------------------------------------------------------------------ isolated stages: original vs optimized
    if not args.skip_legacy and args.backend == "torch":
        img = next(synthetic_frames(args.img_size, args.persons, 1))
        bbs = synthetic_bbs(args.img_size, args.persons)
        pose_net = pipeline.pose_estimator.run.module
        detector = pipeline.detector.run.module
        yolo_size = YoloV4Config().model.input_size

        def legacy_pose():
            with torch.no_grad():
                pedrec_recognizer(pose_net, PedRecNet50Config(), img, bbs, device)

        def optimized_pose():
            with torch.inference_mode():
                pipeline.pose_estimator(gpu_ops.frame_to_tensor(img, device), bbs)

        def legacy_detect():
            sized = cv2.resize(img, (yolo_size.width, yolo_size.height))
            with torch.no_grad():
                do_detect(detector, sized, args.img_size, 0.4, 0.6, device, logger)

        def optimized_detect():
            with torch.inference_mode():
                pipeline.detector(gpu_ops.frame_to_tensor(img, device), args.img_size, 0.4, 0.6)

        rows = []
        for name, legacy, optimized in [("detector (YoloV4)", legacy_detect, optimized_detect),
                                        (f"pose (PedRecNet, {args.persons} persons)", legacy_pose, optimized_pose)]:
            t_legacy = timeit(legacy, device, args.repeats)
            t_new = timeit(optimized, device, args.repeats)
            rows.append((name, t_legacy, t_new))
        print("\nStage comparison (median, ms)")
        print(f"{'stage':38} {'original':>10} {'optimized':>10} {'speedup':>8}")
        for name, t_legacy, t_new in rows:
            print(f"{name:38} {t_legacy * 1000:10.1f} {t_new * 1000:10.1f} {t_legacy / t_new:7.2f}x")
        print("(the original legacy tracker additionally ran PedRecNet a second time for tracked but undetected "
              "persons; the optimized pipeline reuses the first batch)")

    # ------------------------------------------------------------------ full pipeline
    frames = video_frames(args.video, args.img_size, args.frames) if args.video \
        else synthetic_frames(args.img_size, args.persons, args.frames)
    if args.random_weights and not args.video:
        # a random detector finds nobody: still run it (timing), but continue with the known person bbs
        detect = pipeline.detect
        frame_counter = iter(range(10 ** 9))

        def detect_with_synthetic_persons(frame, tracked_humans):
            detect(frame, tracked_humans)
            shift = 3 * next(frame_counter)
            return [bb + np.array([shift, 0, 0, 0, 0, 0], dtype=np.float32)
                    for bb in synthetic_bbs(args.img_size, args.persons)], []
        pipeline.detect = detect_with_synthetic_persons
    timings = defaultdict(list)
    count = 0
    for nr, img in enumerate(frames, start=1):
        sync(device)
        result = pipeline.process(nr, img)
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
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
