"""
Checks the data root before training: which models and datasets are downloaded / converted / missing, whether the
images the training reads exist (sampled at exactly the dataframe rows the training uses) and what to do next.

    mise run data:check                 # or: python pedrec/tools/datasets/check_datasets.py [--samples 200]

Exit code 0 when everything needed by ``train:pedrec`` (default: the published stage chain up to
p2d3d_c_o_h36m_sim_mebow) is present, 1 otherwise. Image samples of large dataframes
are cached in ``<datasets>/.check_cache.json`` (invalidated when the dataframe changes).
"""
import sys

sys.path.append('.')

import argparse
import glob
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from pedrec.configs import default_paths
from pedrec.configs.default_paths import get_data_root
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths

OK, PARTIAL, MISSING, OPTIONAL, INFO = "ok", "partial", "missing", "optional", "info"
SYMBOLS = {OK: "✔", PARTIAL: "◐", MISSING: "✖", OPTIONAL: "○", INFO: "·"}
COLORS = {OK: "32", PARTIAL: "33", MISSING: "31", OPTIONAL: "36", INFO: "37"}

SIM_PATTERN = "view_{cam_name}-frame_{id}.{type}"  # SIM-ROM / SIM-Circle / SIM-C01 (dataset_configs.py)
IMG_PATTERN = "img_{id}.{type}"  # Human3.6m
H36M_STEPS = {"train": 10, "val": 64}  # rows the training / validation loads (systematic subsampling)
COCO_IMAGES = {"train": 118287, "val": 5000}


@dataclass
class Item:
    name: str
    status: str
    detail: str = ""
    fix: str = ""


@dataclass
class Group:
    title: str
    items: List[Item] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", fix: str = "") -> Item:
        item = Item(name, status, detail, fix)
        self.items.append(item)
        return item

    @property
    def complete(self) -> bool:
        return all(item.status in (OK, INFO, OPTIONAL) for item in self.items)


# ---------------------------------------------------------------------------------------------------------- helpers
def count_files(directory: str, suffixes: Tuple[str, ...] = (), limit: Optional[int] = None) -> int:
    if not os.path.isdir(directory):
        return 0
    count = 0
    with os.scandir(directory) as entries:
        for entry in entries:
            if entry.is_file() and (not suffixes or entry.name.lower().endswith(suffixes)):
                count += 1
                if limit and count >= limit:
                    break
    return count


def file_status(path: str) -> bool:
    return os.path.isfile(path)


class ImageSampler:
    """Image paths of n dataframe rows (evenly spread over the rows the training reads), cached per dataframe."""

    def __init__(self, cache_path: str, samples: int):
        self.cache_path, self.samples = cache_path, samples
        try:
            with open(cache_path) as f:
                self.cache = json.load(f)
        except (OSError, ValueError):
            self.cache = {}
        self.changed = False

    def paths(self, df_path: str, pattern: str, step: int = 1) -> List[str]:
        stat = os.stat(df_path)
        key = f"{os.path.abspath(df_path)}|{stat.st_size}|{stat.st_mtime_ns}|{pattern}|{step}|{self.samples}"
        if key not in self.cache:
            import pandas as pd
            print(f"  reading {df_path} ...", flush=True)
            df = pd.read_pickle(df_path)
            rows = np.arange(0, len(df), step)
            rows = rows[np.unique(np.linspace(0, len(rows) - 1, min(self.samples, len(rows))).astype(int))] \
                if len(rows) else rows
            sample = df.iloc[rows]
            paths = []
            for img_dir, img_id, img_type in zip(sample["img_dir"].astype(str), sample["img_id"], sample["img_type"]):
                cam_name = os.path.basename(os.path.normpath(img_dir))
                paths.append(os.path.join(img_dir, pattern.format(id=str(int(img_id)).zfill(5), type=img_type,
                                                                  cam_name=cam_name)))
            self.cache = {k: v for k, v in self.cache.items() if not k.startswith(os.path.abspath(df_path) + "|")}
            self.cache[key] = paths
            self.changed = True
        return self.cache[key]

    def check(self, root: str, df_path: str, pattern: str, step: int = 1) -> Tuple[int, int, Optional[str]]:
        """(present, checked, first missing path)"""
        paths = self.paths(df_path, pattern, step)
        missing = [p for p in paths if not os.path.isfile(os.path.join(root, p))]
        return len(paths) - len(missing), len(paths), missing[0] if missing else None

    def save(self):
        if self.changed:
            try:
                with open(self.cache_path, "w") as f:
                    json.dump(self.cache, f)
            except OSError:
                pass


def check_df_with_images(group: Group, name: str, root: str, filename: str, sampler: ImageSampler, pattern: str,
                         step: int, df_fix: str, image_fix: str, required: bool = True) -> bool:
    """Dataframe + sampled images; returns True if both are complete."""
    df_path = os.path.join(root, filename)
    if not os.path.isfile(df_path):
        group.add(f"{name} dataframe", MISSING if required else OPTIONAL, df_path, df_fix)
        return False
    group.add(f"{name} dataframe", OK, df_path)
    present, checked, first_missing = sampler.check(root, df_path, pattern, step)
    detail = f"{present}/{checked} sampled images present" + (f" (every {step}th row)" if step > 1 else "")
    if checked and present == checked:
        group.add(f"{name} images", OK, detail)
        return True
    status = PARTIAL if present else (MISSING if required else OPTIONAL)
    group.add(f"{name} images", status, f"{detail}, e.g. missing {os.path.join(root, first_missing)}", image_fix)
    return False


# ---------------------------------------------------------------------------------------------------------- checks
def check_models(data_root: str) -> Group:
    group = Group("Pretrained models (mise run download:models)")
    fix = "mise run download:models"
    for name, path, note, required in (
            ("pose-resnet 50 (COCO)", default_paths.POSE_RESNET_WEIGHTS,
             "initialization of the first stage p2d_coco_only", True),
            ("YoloV4 detector", default_paths.YOLO_V4_WEIGHTS, "demo / inference", False),
            ("PedRecNet (published)", default_paths.PEDREC_NET_WEIGHTS, "demo / inference", False),
            ("EHPI3D (published)", default_paths.EHPI3D_WEIGHTS, "demo / inference", False)):
        path = os.path.join(data_root, path)
        group.add(name, OK if os.path.isfile(path) else (MISSING if required else OPTIONAL), f"{path} ({note})", fix)
    return group


def check_pedrec_training(paths, datasets: str, sampler: ImageSampler) -> Group:
    group = Group("PedRecNet training (train:pedrec, stage chain up to p2d3d_c_o_h36m_sim_mebow)")
    # COCO
    coco = paths.coco_dir
    annotations = [os.path.join(coco, "annotations", f"person_keypoints_{s}2017.json") for s in ("train", "val")]
    missing = [a for a in annotations if not os.path.isfile(a)]
    group.add("COCO keypoint annotations", MISSING if missing else OK, missing[0] if missing else
              os.path.join(coco, "annotations"), "mise run download:datasets:coco --parts annotations")
    for split, expected in COCO_IMAGES.items():
        n = count_files(os.path.join(coco, f"{split}2017"), (".jpg",))
        status = OK if n >= expected else (PARTIAL if n else MISSING)
        group.add(f"COCO {split}2017 images", status, f"{n}/{expected} images",
                  f"mise run download:datasets:coco --parts {split}")
    mebow = [os.path.join(coco, "annotations", f"{s}_hoe.json") for s in ("train", "val")]
    missing = [m for m in mebow if not os.path.isfile(m)]
    group.add("MEBOW orientation labels (COCO)", MISSING if missing else OK,
              (f"missing {missing[0]}; needed by the *_mebow stages (orientation)" if missing
               else "train_hoe.json, val_hoe.json"),
              "mise run download:datasets:info mebow  (by e-mail from the MEBOW authors)")
    # Human3.6m
    for split, root, filename in (("train", paths.h36m_train_dir, paths.h36m_train_filename),
                                  ("val", paths.h36m_val_dir, paths.h36m_val_filename)):
        videos = sum(count_files(os.path.join(d, "Videos"), (".mp4",)) for d in glob.glob(os.path.join(root, "S*")))
        images_ok = check_df_with_images(group, f"Human3.6m {split}", root, filename, sampler, IMG_PATTERN,
                                         H36M_STEPS[split], "mise run download:datasets:pedrec --parts h36m",
                                         "mise run data:h36m:images")
        if images_ok:
            group.add(f"Human3.6m {split} videos", INFO, f"{videos} videos (only needed to extract the images)")
        else:
            group.add(f"Human3.6m {split} videos", OK if videos else MISSING,
                      f"{videos} videos below {os.path.join(root, 'S*', 'Videos')}",
                      "mise run download:datasets:info h36m  (registration, only the \"Videos\"), then mise run "
                      "data:h36m:images")
    # SIM
    check_df_with_images(group, "SIM-ROM (train)", paths.sim_train_dir, paths.sim_train_filename, sampler, SIM_PATTERN,
                         1, "mise run download:datasets:pedrec --parts rom",
                         "mise run download:datasets:pedrec --parts rom (archive incomplete?)")
    check_df_with_images(group, "SIM-Circle (val)", paths.sim_val_dir, paths.sim_val_filename, sampler, SIM_PATTERN,
                         10, "mise run download:datasets:pedrec --parts circle",
                         "mise run download:datasets:pedrec --parts circle (archive incomplete?)")
    return group


def check_action_recognition(paths, sampler: ImageSampler) -> Group:
    group = Group("Action recognition (train:ehpi3d)")
    c01_fix = "mise run download:datasets:pedrec --parts c01"
    train_df = os.path.join(paths.sim_c01_dir, paths.sim_c01_filename)
    val_df = os.path.join(paths.sim_c01_val_dir, paths.sim_c01_val_filename)
    for name, path in (("SIM-C01 train dataframe", train_df), ("SIM-C01 val dataframe", val_df)):
        group.add(name, OK if os.path.isfile(path) else MISSING, path, c01_fix)
    for split, directory, filename in (("train", paths.sim_c01_dir, paths.sim_c01_results_filename),
                                       ("val", paths.sim_c01_val_dir, paths.sim_c01_val_results_filename)):
        path = os.path.join(directory, filename)
        group.add(f"SIM-C01 {split} PedRecNet results", OK if os.path.isfile(path) else MISSING,
                  f"{path} (input of the gt_pred variants)", "mise run download:datasets:pedrec --parts c01-results")
    if os.path.isfile(train_df):
        present, checked, _ = sampler.check(paths.sim_c01_dir, train_df, SIM_PATTERN, 1)
        group.add("SIM-C01 images", OK if checked and present == checked else INFO,
                  f"{present}/{checked} sampled images present (not published; only needed to regenerate the results "
                  "with train:ehpi3d:data)")
    return group


def check_evaluation(paths) -> Group:
    group = Group("Optional")
    tud = count_files(paths.tud_dir, (".png", ".jpg"), limit=10)
    group.add("TUD multiview pedestrians", OK if tud else OPTIONAL,
              paths.tud_dir if tud else "only for the *_tud stages and eval:tud-orientation",
              "mise run download:datasets:info tud")
    return group


def check_weights(paths) -> Group:
    group = Group("Training checkpoints (results of train:pedrec / train:ehpi3d)")
    stages = sorted(os.path.basename(f)[len("experiment_pedrec_"):-len("_0.pth")]
                    for f in glob.glob(os.path.join(paths.output_dir, "experiment_pedrec_*_0.pth")))
    group.add("PedRecNet stages", OK if stages else INFO,
              ", ".join(stages) if stages else "none trained yet (mise run train:pedrec:chain)")
    actions = glob.glob(os.path.join(paths.ehpi3d_output_dir, "*.pth"))
    group.add("EHPI3D", OK if actions else INFO,
              f"{len(actions)} checkpoints in {paths.ehpi3d_output_dir}" if actions else "none trained yet "
              "(mise run train:ehpi3d)")
    return group


# ---------------------------------------------------------------------------------------------------------- output
def _fmt(status: str, text: str, color: bool) -> str:
    symbol = SYMBOLS[status]
    return f"\033[{COLORS[status]}m{symbol}\033[0m {text}" if color else f"{symbol} {text}"


def merge_steps(fixes: List[str]) -> List[str]:
    """Unique fixes in order; several "--parts" of the same download task become one command."""
    steps: List[str] = []
    parts: Dict[str, List[str]] = {}
    for fix in fixes:
        if " --parts " in fix:
            task, part = fix.split(" --parts ", 1)
            if task not in parts:
                parts[task] = []
                steps.append(task)
            if part not in parts[task]:
                parts[task].append(part)
        elif fix not in steps:
            steps.append(fix)
    all_parts = {"mise run download:datasets:coco": 3, "mise run download:datasets:pedrec": 5}
    return [step if step not in parts or len(parts[step]) == all_parts.get(step) else
            f"{step} --parts {' '.join(parts[step])}" for step in steps]


def print_report(groups: List[Group], summary: List[str], steps: List[str], recommendations: List[str],
                 color: bool, data_root: str = ""):
    def short(text: str) -> str:  # paths relative to the data root
        for prefix in {os.path.abspath(data_root), data_root} - {""}:
            text = text.replace(prefix.rstrip(os.sep) + os.sep, "")
        return text
    for group in groups:
        print(f"\n{group.title}")
        for item in group.items:
            print("  " + _fmt(item.status, f"{item.name:36} {short(item.detail)}", color))
            if item.fix and item.status in (MISSING, PARTIAL):
                print(f"  {'':38}-> {short(item.fix)}")
    print("\nSummary")
    for line in summary:
        print("  " + line)
    if steps:
        print("\nNext steps")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {short(step)}")
    if recommendations:
        print("\nRecommendations")
        for line in recommendations:
            print(f"  - {line}")


def run(data_root: Optional[str] = None, samples: int = 200, color: Optional[bool] = None) -> int:
    data_root = get_data_root(data_root)
    paths = get_experiment_paths(data_root)
    datasets = os.path.join(data_root, "datasets")
    print(f"Data root: {os.path.abspath(data_root)}")
    sampler = ImageSampler(os.path.join(datasets, ".check_cache.json") if os.path.isdir(datasets)
                           else os.devnull, samples)
    models = check_models(data_root)
    pedrec = check_pedrec_training(paths, datasets, sampler)
    actions = check_action_recognition(paths, sampler)
    groups = [models, pedrec, actions, check_evaluation(paths), check_weights(paths)]
    sampler.save()

    color = sys.stdout.isatty() if color is None else color
    ready_pedrec = models.complete and pedrec.complete
    ready_actions = actions.complete

    def line(ready: bool, task: str, note: str = "") -> str:
        return _fmt(OK if ready else MISSING, f"{task:14} {'ready' if ready else 'not ready'}{note}", color)
    summary = [line(ready_pedrec, "train:pedrec"), line(ready_actions, "train:ehpi3d")]
    steps = merge_steps([item.fix for group in groups for item in group.items
                         if item.status in (MISSING, PARTIAL) and item.fix])
    recommendations = ["Additional 3D datasets (MPI-INF-3DHP, Fit3D, AIST++, AMASS) and the temporal 3D lifter are "
                       "available on the v2 branch (same data layout, the datasets here can be reused there)"]
    total, used, free = shutil.disk_usage(datasets if os.path.isdir(datasets) else data_root if os.path.isdir(
        data_root) else ".")
    recommendations.append(f"Free disk space: {free / 1e9:.0f} GB")
    print_report(groups, summary, steps, recommendations, color, data_root)
    return 0 if ready_pedrec else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="Data root (default: $PEDREC_DATA_DIR or 'data').")
    parser.add_argument("--samples", type=int, default=200, help="Sampled images per dataframe (default 200).")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)
    sys.exit(run(args.data_dir, args.samples, False if args.no_color else None))


if __name__ == "__main__":
    main()
