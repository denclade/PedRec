"""
Downloads the training datasets into ``$PEDREC_DATA_DIR/datasets`` (default ``data/datasets``), see the README
section "Datasets" and ``mise run download:datasets:<name>``.

Downloads are resumable (``*.part`` files), existing files / directories are skipped. Datasets which require a
registration (Human3.6m, Fit3D, AMASS, MEBOW, TUD) can not be downloaded automatically, ``info <name>`` prints where to
get them and where to put them.

    python pedrec/tools/datasets/download_datasets.py list
    python pedrec/tools/datasets/download_datasets.py pedrec            # SIM-ROM, SIM-Circle, SIM-C01, H36M dataframes
    python pedrec/tools/datasets/download_datasets.py coco
    python pedrec/tools/datasets/download_datasets.py h36m-images       # frames of the registered H36M videos
    python pedrec/tools/datasets/download_datasets.py 3dhp --accept-license
    python pedrec/tools/datasets/download_datasets.py aistpp --accept-terms --views c01 c05
    python pedrec/tools/datasets/download_datasets.py info amass
"""
import sys

sys.path.append('.')

import argparse
import os
import re
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence

from pedrec.configs.default_paths import get_data_root

DENNISNOTES = "https://dennisnotes.com/files/pedrec/datasets"

INFO: Dict[str, str] = {
    "pedrec": """\
SIM-ROM (ROMb), SIM-Circle (RT3DValidate), SIM-C01 and the Human3.6m dataframes of PedRec (dennisnotes.com):
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
    1. Register / log in, download per subject (S1, S5, S6, S7, S8 -> Human3.6m/train, S9, S11 -> Human3.6m/val):
       Videos, "D2 Positions", "D3 Positions mono" (MyPoseFeatures) and the bounding boxes (MySegmentsMat), extracted
       as Human3.6m/<split>/<subject>/{Videos,MyPoseFeatures,MySegmentsMat}.
    2. mise run download:datasets:pedrec   (dataframes h36m_{train,val}_pedrec.pkl)
    3. mise run data:h36m:images           (extracts the frames used by the dataframes)
       Alternatively regenerate everything from the raw data: mise run data:convert:h36m""",
    "3dhp": """\
MPI-INF-3DHP (Mehta et al., 3DV 2017, https://vcai.mpi-inf.mpg.de/3dhp-dataset), non-commercial research license:
    Read the license on the project page, then: mise run download:datasets:3dhp --accept-license
    (downloads the official download scripts and runs them, needs bash + wget, ~25 GB for all cameras;
    --cameras 0 2 5 8 limits the views), then mise run data:convert:3dhp""",
    "fit3d": """\
Fit3D (Fieraru et al., AAAI 2021, https://fit3d.imar.ro), registration + non-commercial license:
    Register, download fit3d_train.zip and extract it to datasets/Fit3D (-> datasets/Fit3D/train/s03/...).
    HumanSC3D / CHI3D (https://sc3d.imar.ro, https://ci3d.imar.ro) have the same layout:
    python pedrec/tools/datasets/convert_fit3d.py --root data/datasets/CHI3D --name CHI3D
    then mise run data:convert:fit3d""",
    "aistpp": """\
AIST++ (Li et al., ICCV 2021, https://google.github.io/aistplusplus_dataset), annotations CC BY 4.0, videos of the
AIST Dance Video Database (research use, terms: https://aistdancedb.ongaaccel.jp/terms_of_use/):
    mise run download:datasets:aistpp --accept-terms [--views c01 c05 c09] [--max-sequences 200]
    then mise run data:convert:aistpp""",
    "amass": """\
AMASS (Mahmood et al., ICCV 2019, https://amass.is.tue.mpg.de), registration + non-commercial license:
    1. Register, download the "SMPL+H G" archives of the wanted datasets (recommended: MPI_Limits (range of motion),
       CMU, BMLmovi, KIT, HDM05, TotalCapture, ...) and extract them to datasets/AMASS (-> AMASS/MPI_Limits/...).
    2. Body model: "Extended SMPL+H model" from https://mano.is.tue.mpg.de (or the AMASS download page), extracted to
       models/body_models/smplh (-> smplh/{male,female,neutral}/model.npz).
    3. mise run data:convert:amass   (sequences for the 3D lifter only, AMASS has no images)""",
    "tud": """\
TUD multiview pedestrians (Andriluka et al., CVPR 2010), orientation evaluation only (eval:tud-orientation):
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/people-detection-pose-estimation-and-tracking/monocular-3d-pose-estimation-and-tracking-by-detection
    Extract to datasets/cvpr10_multiview_pedestrians.""",
}


def log(msg: str):
    print(msg, flush=True)


def _fetch(url: str, path: str, retries: int = 3):
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
                        if time.time() - last_report > 5:
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


def download(urls: Sequence[str], path: str) -> str:
    """Downloads the first working url of urls to path (skipped if path exists)."""
    if os.path.isfile(path):
        log(f"exists: {path}")
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    errors = []
    for url in [urls] if isinstance(urls, str) else urls:
        log(f"download {url} -> {path}")
        try:
            _fetch(url, path)
            return path
        except Exception as e:  # try the next mirror
            errors.append(f"{url}: {e}")
    raise RuntimeError("Download failed:\n  " + "\n  ".join(errors))


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


def extract_h36m_images(datasets: str, splits: List[str], quality: int = 92):
    """Extracts the frames referenced by the H36M dataframes from the (registered) Human3.6m videos."""
    from pedrec.tools.datasets.pedrec_df_writer import extract_frames
    from pedrec.utils.pandas_helper import read_pedrec_df
    for split in splits:
        base = os.path.join(datasets, "Human3.6m", split)
        df_path = os.path.join(base, f"h36m_{split}_pedrec.pkl")
        if not os.path.isfile(df_path):
            raise FileNotFoundError(f"{df_path} missing, run mise run download:datasets:pedrec first")
        df = read_pedrec_df(df_path)
        groups = df.groupby(df["img_dir"].astype(str), observed=True)["img_id"]
        for i, (img_dir, ids) in enumerate(groups):
            subject, _, name = img_dir.replace("\\", "/").split("/")
            video = os.path.join(base, subject, "Videos", f"{name}.mp4")
            if not os.path.isfile(video):
                log(f"missing video {video}")
                continue
            frames = ids.to_numpy().astype(int) - 1
            written = extract_frames(video, os.path.join(base, img_dir), frames, quality)
            if len(written) != len(set(frames)):
                log(f"  {img_dir}: only {len(written)} of {len(set(frames))} frames in the video")
            if (i + 1) % 20 == 0:
                log(f"{split}: {i + 1}/{groups.ngroups} videos")


def get_3dhp(datasets: str, accept_license: bool, subjects: List[int], cameras: List[int], keep_archives: bool):
    if not accept_license:
        raise SystemExit("MPI-INF-3DHP is licensed for non-commercial research only. Read the license on "
                         "https://vcai.mpi-inf.mpg.de/3dhp-dataset/ and pass --accept-license.")
    root = os.path.abspath(os.path.join(datasets, "MPI-INF-3DHP"))
    tools = os.path.join(root, "download_tools")
    download_and_extract(["https://vcai.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip",
                          "http://gvv.mpi-inf.mpg.de/3dhp-dataset/mpi_inf_3dhp.zip"], tools, "conf.ig", keep_archives)
    conf_path = os.path.join(tools, "conf.ig")
    conf = open(conf_path).read()
    if "ready_to_download" not in conf:
        raise SystemExit(f"Unknown format of {conf_path}, edit it and run get_dataset.sh / get_testset.sh manually.")
    conf = re.sub(r"(?m)^ready_to_download=.*$", "ready_to_download=1", conf)
    conf = re.sub(r"(?m)^destination=.*$", f"destination='{root}/'", conf)
    conf = re.sub(r"(?m)^subjects=.*$", f"subjects=({' '.join(map(str, subjects))})", conf)
    if cameras:
        conf = re.sub(r"(?m)^cameras=.*$", f"cameras=({' '.join(map(str, cameras))})", conf)
    open(conf_path, "w").write(conf)
    for script in ("get_dataset.sh", "get_testset.sh"):
        if os.path.isfile(os.path.join(tools, script)):
            log(f"run {script}")
            subprocess.run(["bash", script], cwd=tools, check=True)
    log("done, convert with: mise run data:convert:3dhp")


AISTPP_ANNOTATIONS = "https://storage.googleapis.com/aist_plusplus_public/20210308/fullset.zip"
AISTPP_VIDEO_LIST = "https://storage.googleapis.com/aist_plusplus_public/20121228/video_list.txt"
AISTPP_VIDEOS = "https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M"


def get_aistpp(datasets: str, accept_terms: bool, views: List[str], max_sequences: Optional[int], workers: int,
               keep_archives: bool):
    if not accept_terms:
        raise SystemExit("The AIST++ videos are part of the AIST Dance Video Database, read the terms of use "
                         "(https://aistdancedb.ongaaccel.jp/terms_of_use/) and pass --accept-terms.")
    root = os.path.join(datasets, "AIST++")
    annotations = os.path.join(root, "annotations")
    download_and_extract(AISTPP_ANNOTATIONS, annotations, "keypoints3d", keep_archives)
    ignore_path = os.path.join(annotations, "ignore_list.txt")
    ignore = set(open(ignore_path).read().split()) if os.path.isfile(ignore_path) else set()
    sequences = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(annotations, "keypoints3d")))
    sequences = [s for s in sequences if s not in ignore][:max_sequences]
    video_list = download(AISTPP_VIDEO_LIST, os.path.join(root, "video_list.txt"))
    available = set(open(video_list).read().split())
    names = [s.replace("cAll", view) for s in sequences for view in views]
    names = [n for n in names if n in available]
    video_dir = os.path.join(root, "videos")
    log(f"{len(names)} videos ({len(sequences)} sequences x {len(views)} views) -> {video_dir}")

    def get(name):
        try:
            download(f"{AISTPP_VIDEOS}/{name}.mp4", os.path.join(video_dir, f"{name}.mp4"))
        except Exception as e:
            log(f"failed: {name}: {e}")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(get, names))
    log("done, convert with: mise run data:convert:aistpp")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets-dir", default=None, help="Default: $PEDREC_DATA_DIR/datasets")
    parser.add_argument("--keep-archives", action="store_true", help="Keep the archives after extraction.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="Overview of the supported datasets")
    info = sub.add_parser("info", help="How to get a dataset")
    info.add_argument("name", choices=sorted(INFO))
    p = sub.add_parser("pedrec", help="PedRec datasets from dennisnotes.com")
    p.add_argument("--parts", nargs="*", default=["rom", "circle", "c01", "h36m"],
                   choices=["rom", "circle", "c01", "h36m"])
    p = sub.add_parser("coco", help="COCO 2017 images + annotations")
    p.add_argument("--parts", nargs="*", default=["annotations", "val", "train"], choices=["annotations", "val", "train"])
    p = sub.add_parser("h36m-images", help="Extract the frames of the H36M dataframes from the registered videos")
    p.add_argument("--splits", nargs="*", default=["train", "val"], choices=["train", "val"])
    p = sub.add_parser("3dhp", help="MPI-INF-3DHP via the official download scripts")
    p.add_argument("--accept-license", action="store_true")
    p.add_argument("--subjects", nargs="*", type=int, default=list(range(1, 9)))
    p.add_argument("--cameras", nargs="*", type=int, default=[0, 1, 2, 4, 5, 6, 7, 8],
                   help="Chest height cameras by default.")
    p = sub.add_parser("aistpp", help="AIST++ annotations + AIST Dance DB videos")
    p.add_argument("--accept-terms", action="store_true")
    p.add_argument("--views", nargs="*", default=[f"c{i:02d}" for i in range(1, 10)])
    p.add_argument("--max-sequences", type=int, default=None)
    p.add_argument("--workers", type=int, default=4)
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
        extract_h36m_images(datasets, args.splits)
    elif args.command == "3dhp":
        get_3dhp(datasets, args.accept_license, args.subjects, args.cameras, args.keep_archives)
    elif args.command == "aistpp":
        get_aistpp(datasets, args.accept_terms, args.views, args.max_sequences, args.workers, args.keep_archives)


if __name__ == "__main__":
    main()
