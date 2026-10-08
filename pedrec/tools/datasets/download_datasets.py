"""
Downloads the training datasets into ``$PEDREC_DATA_DIR/datasets`` (default ``data/datasets``), see the README
section "Datasets" and ``mise run download:datasets:<name>``.

Downloads are resumable (``*.part`` files), existing files / directories are skipped. Datasets which require a
registration (Human3.6m, MEBOW, TUD) can not be downloaded automatically, ``info <name>`` prints where to
get them and where to put them.

    python pedrec/tools/datasets/download_datasets.py list
    python pedrec/tools/datasets/download_datasets.py pedrec            # SIM-ROM, SIM-Circle, SIM-C01, H36M dataframes,
                                                                        # SIM-C01 results of the published PedRecNet
    python pedrec/tools/datasets/download_datasets.py coco
    python pedrec/tools/datasets/download_datasets.py h36m-images       # frames of the registered H36M videos
    python pedrec/tools/datasets/download_datasets.py info h36m
"""
import sys

sys.path.append('.')

import argparse
import os
import re
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from typing import Dict, List, Optional, Sequence

from pedrec.configs.default_paths import get_data_root

DENNISNOTES = "https://dennisnotes.com/files/pedrec/datasets"
RESULT_DFS = "https://dennisnotes.com/files/pedrec/result_dfs"
# PedRecNet (published p2d3d_c_o_h36m_sim_mebow) results on SIM-C01, input of the EHPI3D training (see
# experiment_path_helper.py); regenerating them (train:ehpi3d:data) needs the unpublished SIM-C01 images
C01_RESULTS = ["C01F_train_pred_df_experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_allframes.pkl",
               "C01F_pred_df_experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_allframes.pkl"]
PEDREC_PARTS = ["rom", "circle", "c01", "c01-results", "h36m"]

INFO: Dict[str, str] = {
    "pedrec": """\
SIM-ROM (ROMb), SIM-Circle (RT3DValidate), SIM-C01 (+ the results of the published PedRecNet on it, input of the
EHPI3D training) and the Human3.6m dataframes of PedRec (dennisnotes.com):
    mise run download:datasets:pedrec""",
    "coco": """\
COCO 2017 keypoints (CC BY 4.0 annotations, Flickr image terms), ~19 GB:
    mise run download:datasets:coco
Body orientation labels: see 'info mebow'.""",
    "mebow": """\
MEBOW body orientation labels for COCO (Wu et al., CVPR 2020, https://github.com/ChenyanWu/MEBOW):
    The annotations are sent on request by e-mail (see the MEBOW README). Put train_hoe.json and val_hoe.json into
    datasets/COCO/annotations/. Needed by the training stage p2d3d_c_o (orientation).""",
    "h36m": """\
Human3.6m (Ionescu et al., TPAMI 2014, http://vision.imar.ro/human3.6m), registration + non-commercial license:
    1. Register / log in, download the "Videos" of the subjects S1, S5, S6, S7, S8 (-> Human3.6m/train) and S9, S11
       (-> Human3.6m/val), extracted as Human3.6m/<split>/<subject>/Videos/*.mp4. With the PedRec dataframes only
       the videos are needed ("D2 Positions", "D3 Positions mono" and the bounding boxes (MySegmentsMat) only to
       regenerate the dataframes with data:convert:h36m).
    2. mise run download:datasets:pedrec   (dataframes h36m_{train,val}_pedrec.pkl)
    3. mise run data:h36m:images           (extracts the frames the training uses: every 10th training frame
       (~156k images), every 64th validation frame (~8.5k); --val-step 1 for all validation frames (~0.5M images,
       needed by results:h36m), --train-step 1 for all training frames (~1.5M images))""",
    "tud": """\
TUD multiview pedestrians (Andriluka et al., CVPR 2010), orientation evaluation only (eval:tud-orientation):
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/people-detection-pose-estimation-and-tracking/monocular-3d-pose-estimation-and-tracking-by-detection
    Extract to datasets/cvpr10_multiview_pedestrians.""",
}


def log(msg: str):
    print(msg, flush=True)


def _fetch(url: str, path: str, retries: int = 3, quiet: bool = False):
    part = path + ".part"
    for attempt in range(retries):
        offset = os.path.getsize(part) if os.path.isfile(part) else 0
        request = urllib.request.Request(url, headers={"User-Agent": "pedrec-dataset-downloader"})
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:  # server ignores the range -> restart
                    offset = 0
                total = response.headers.get("Content-Length")
                total = int(total) + offset if total else None
                done, last_report = offset, 0.0
                with open(part, "ab" if offset else "wb") as f:
                    while chunk := response.read(1 << 20):
                        f.write(chunk)
                        done += len(chunk)
                        if not quiet and time.time() - last_report > 5:
                            last_report = time.time()
                            progress = f"{done / total * 100:5.1f}% of {total / 1e9:.2f} GB" if total \
                                else f"{done / 1e9:.2f} GB"
                            log(f"  {os.path.basename(path)}: {progress}")
            os.replace(part, path)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416:  # range not satisfiable: already complete
                os.replace(part, path)
                return
            if e.code in (401, 403, 404):
                raise
            error = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = e
        log(f"  retry {attempt + 1}/{retries} after error: {error}")
        time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Download failed: {url}")


def download(urls: Sequence[str], path: str, quiet: bool = False) -> str:
    """Downloads the first working url of urls to path (skipped if path exists)."""
    if os.path.isfile(path):
        if not quiet:
            log(f"exists: {path}")
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    errors = []
    for url in [urls] if isinstance(urls, str) else urls:
        if not quiet:
            log(f"download {url} -> {path}")
        try:
            _fetch(url, path, quiet=quiet)
            return path
        except Exception as e:  # try the next mirror
            errors.append(f"{url}: {e}")
    raise RuntimeError("Download failed:\n  " + "\n  ".join(errors))


def _remove_os_junk(directory: str):
    """macOS metadata in archives (__MACOSX/, ._<name>, .DS_Store)."""
    shutil.rmtree(os.path.join(directory, "__MACOSX"), ignore_errors=True)
    for parent, _, files in os.walk(directory):
        for name in files:
            if name.startswith("._") or name == ".DS_Store":
                os.remove(os.path.join(parent, name))


def extract(archive: str, target_dir: str, keep_archive: bool = False):
    """
    Extracts archive into target_dir. If the archive contains a single top level directory, its content is moved to
    target_dir (archives with and without the folder in it result in the same layout).
    """
    tmp = target_dir.rstrip("/\\") + ".extracting"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    log(f"extract {archive} -> {target_dir}")
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
    elif archive.endswith(".7z"):
        import py7zr
        with py7zr.SevenZipFile(archive) as z:
            z.extractall(tmp)
    elif re.search(r"\.tar(\.\w+)?$|\.tgz$", archive):
        with tarfile.open(archive) as t:
            t.extractall(tmp, filter="data")
    else:
        raise ValueError(f"Unknown archive type: {archive}")
    _remove_os_junk(tmp)
    content = os.listdir(tmp)
    source = os.path.join(tmp, content[0]) if len(content) == 1 and os.path.isdir(os.path.join(tmp, content[0])) \
        else tmp
    os.makedirs(target_dir, exist_ok=True)
    for name in os.listdir(source):
        destination = os.path.join(target_dir, name)
        if os.path.exists(destination):
            log(f"  keeping existing {destination}")
            continue
        shutil.move(os.path.join(source, name), destination)
    shutil.rmtree(tmp, ignore_errors=True)
    if not keep_archive:
        os.remove(archive)


def download_and_extract(urls: Sequence[str], target_dir: str, marker: str, keep_archive: bool = False):
    """Skipped when target_dir/marker exists."""
    if os.path.exists(os.path.join(target_dir, marker)):
        log(f"exists: {os.path.join(target_dir, marker)}")
        return
    first = urls if isinstance(urls, str) else urls[0]
    archive = os.path.join(os.path.dirname(target_dir.rstrip("/\\")), os.path.basename(first.split("?")[0]))
    extract(download(urls, archive), target_dir, keep_archive)


# ---------------------------------------------------------------------------------------------------------- datasets
def get_pedrec(datasets: str, parts: List[str], keep_archives: bool):
    if "h36m" in parts:
        for split in ("train", "val"):
            download(f"{DENNISNOTES}/H36M/h36m_{split}_pedrec.pkl",
                     os.path.join(datasets, "Human3.6m", split, f"h36m_{split}_pedrec.pkl"))
        log("Human3.6m images: see 'info h36m' (registration needed), then mise run data:h36m:images")
    if "rom" in parts:
        download_and_extract(f"{DENNISNOTES}/ROMb.7z", os.path.join(datasets, "ROMb"), "rt_rom_01b.pkl",
                             keep_archives)
    if "circle" in parts:
        download_and_extract(f"{DENNISNOTES}/RT3DValidate.7z", os.path.join(datasets, "RT3DValidate"),
                             "rt_validate_3d.pkl", keep_archives)
    if "c01" in parts:
        c01 = os.path.join(datasets, "Conti01")
        download(f"{DENNISNOTES}/SIM-C01/rt_conti_01_train_FIN.pkl", os.path.join(c01, "rt_conti_01_train_FIN.pkl"))
        # the training expects the *_FIN name for both splits, the val file was published without the suffix
        download([f"{DENNISNOTES}/SIM-C01/rt_conti_01_val_FIN.pkl", f"{DENNISNOTES}/SIM-C01/rt_conti_01_val.pkl"],
                 os.path.join(c01, "rt_conti_01_val_FIN.pkl"))
    if "c01-results" in parts:
        for filename in C01_RESULTS:
            download(f"{RESULT_DFS}/{filename}", os.path.join(datasets, "Conti01", filename))


def get_coco(datasets: str, parts: List[str], keep_archives: bool):
    coco = os.path.join(datasets, "COCO")
    # part: (url, directory (the top level folder of the archive), marker)
    files = {"annotations": ("annotations/annotations_trainval2017.zip", "annotations", "person_keypoints_val2017.json"),
             "val": ("zips/val2017.zip", "val2017", "000000000139.jpg"),
             "train": ("zips/train2017.zip", "train2017", "000000000009.jpg")}
    for part in parts:
        url, directory, marker = files[part]
        if os.path.exists(os.path.join(coco, directory, marker)):
            log(f"exists: {os.path.join(coco, directory)}")
            continue
        archive = download([f"https://images.cocodataset.org/{url}", f"http://images.cocodataset.org/{url}"],
                           os.path.join(datasets, f"coco_{os.path.basename(url)}"))
        extract(archive, os.path.join(coco, directory), keep_archives)
    if not os.path.isfile(os.path.join(coco, "annotations", "train_hoe.json")):
        log(INFO["mebow"])


H36M_STEPS = {"train": 10, "val": 64}  # systematic subsampling of the training / validation (dataset_configs.py)


def extract_h36m_images(datasets: str, splits: List[str], quality: int = 92, workers: Optional[int] = None,
                        steps: Optional[dict] = None):
    """
    Extracts the frames of the H36M dataframes from the (registered) Human3.6m videos. Only every step-th dataframe
    row is used (as the systematic subsampling of the training: train 10, val 64; 1 = all ~2.1M frames).
    """
    from pedrec.tools.datasets.pedrec_df_writer import FrameExtractor
    from pedrec.utils.pandas_helper import read_pedrec_df
    for split in splits:
        base = os.path.join(datasets, "Human3.6m", split)
        df_path = os.path.join(base, f"h36m_{split}_pedrec.pkl")
        if not os.path.isfile(df_path):
            raise FileNotFoundError(f"{df_path} missing, run mise run download:datasets:pedrec first")
        df = read_pedrec_df(df_path)
        step = (steps or H36M_STEPS)[split]
        df = df.loc[range(0, len(df), step)]  # the rows the training loads (get_subsampled_df)
        groups = df.groupby(df["img_dir"].astype(str), observed=True)["img_id"]
        with FrameExtractor(workers, total=groups.ngroups, desc=f"Human3.6m {split}") as extractor:
            for img_dir, ids in groups:
                subject, _, name = img_dir.replace("\\", "/").split("/")
                video = os.path.join(base, subject, "Videos", f"{name}.mp4")
                frames = ids.to_numpy().astype(int) - 1

                def check(written, img_dir=img_dir, video=video, expected=len(set(frames))):
                    if written is None:
                        extractor.progress.write(f"missing video {video}")
                    elif len(written) != expected:
                        extractor.progress.write(f"  {img_dir}: only {len(written)} of {expected} frames in the video")
                extractor.submit(video, os.path.join(base, img_dir), frames, check, quality=quality)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", default=None, help="Default: $PEDREC_DATA_DIR/datasets")
    parser.add_argument("--keep-archives", action="store_true", help="Keep the archives after extraction.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Overview of the supported datasets")
    info = sub.add_parser("info", help="How to get a dataset")
    info.add_argument("name", choices=sorted(INFO))
    p = sub.add_parser("pedrec", help="PedRec datasets from dennisnotes.com")
    p.add_argument("--parts", nargs="*", default=PEDREC_PARTS, choices=PEDREC_PARTS)
    p = sub.add_parser("coco", help="COCO 2017 images + annotations")
    p.add_argument("--parts", nargs="*", default=["annotations", "val", "train"], choices=["annotations", "val", "train"])
    p = sub.add_parser("h36m-images", help="Extract the frames of the H36M dataframes from the registered videos")
    p.add_argument("--splits", nargs="*", default=["train", "val"], choices=["train", "val"])
    p.add_argument("--train-step", type=int, default=H36M_STEPS["train"],
                   help="Every n-th frame of the training dataframe (default 10 = what the training uses).")
    p.add_argument("--val-step", type=int, default=H36M_STEPS["val"],
                   help="Every n-th frame of the validation dataframe (default 64 = training validation and "
                        "eval:h36m; 1 = all frames, needed by results:h36m).")
    p.add_argument("--workers", type=int, default=None, help="Parallel video decoders (default: CPUs, max 8).")
    args = parser.parse_args(argv)

    datasets = args.datasets_dir or os.path.join(get_data_root(), "datasets")
    if args.command == "list":
        for name, text in INFO.items():
            log(f"[{name}]\n{text}\n")
    elif args.command == "info":
        log(INFO[args.name])
    elif args.command == "pedrec":
        get_pedrec(datasets, args.parts, args.keep_archives)
    elif args.command == "coco":
        get_coco(datasets, args.parts, args.keep_archives)
    elif args.command == "h36m-images":
        extract_h36m_images(datasets, args.splits, workers=args.workers,
                            steps={"train": args.train_step, "val": args.val_step})


if __name__ == "__main__":
    main()
