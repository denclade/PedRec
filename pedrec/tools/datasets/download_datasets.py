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
    then mise run data:convert:aistpp
  Annotations from the GitHub release v1.0 (keypoints3d 876 MB + cameras + splits + ignore_list), videos (1080p,
  60 fps) directly from the AIST Dance DB like the official downloader.py: 1408 sequences x 9 views = 12.7k videos,
  the size of the full set is printed before the download starts (expect a few hundred GB; --views / --max-sequences
  limit it). --annotations-only: only the annotations (enough for the lifter sequences, no images).""",
    "amass": """\
AMASS (Mahmood et al., ICCV 2019, https://amass.is.tue.mpg.de), registration + non-commercial license:
    1. Register, download the "SMPL+H G" archives of the wanted datasets (recommended: MPI_Limits (range of motion),
       CMU, BMLmovi, KIT, HDM05, TotalCapture, ...) and extract them to datasets/AMASS (-> AMASS/MPI_Limits/...).
    2. Body model SMPL+H (= SMPL body + MANO hands, the model AMASS is fitted with; it is hosted on the MANO site,
       not on smpl.is.tue.mpg.de, own registration): https://mano.is.tue.mpg.de -> Download -> "Extended SMPL+H
       model (used in AMASS project)" (smplh.tar.xz, npz files with 16 shape parameters), extracted to
       models/body_models/smplh (-> smplh/{male,female,neutral}/model.npz). Not needed: the "Models & Code" pkl
       files (chumpy) and the DMPLs. The plain SMPL model does not fit (no hand joints, 10 shape parameters).
    3. mise run data:convert:amass   (sequences for the 3D lifter only, AMASS has no images)""",
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
    """macOS metadata in archives (__MACOSX/, ._<name>, .DS_Store), e.g. in the AIST++ annotation archives."""
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


def extract_h36m_images(datasets: str, splits: List[str], quality: int = 92, workers: Optional[int] = None):
    """Extracts the frames referenced by the H36M dataframes from the (registered) Human3.6m videos."""
    from pedrec.tools.datasets.pedrec_df_writer import FrameExtractor
    from pedrec.utils.pandas_helper import read_pedrec_df
    for split in splits:
        base = os.path.join(datasets, "Human3.6m", split)
        df_path = os.path.join(base, f"h36m_{split}_pedrec.pkl")
        if not os.path.isfile(df_path):
            raise FileNotFoundError(f"{df_path} missing, run mise run download:datasets:pedrec first")
        df = read_pedrec_df(df_path)
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


# https://google.github.io/aistplusplus_dataset/download.html: annotations as GitHub release assets, videos from the
# AIST Dance Video Database (as the official downloader.py of google/aistplusplus_api)
AISTPP_RELEASE = "https://github.com/google/aistplusplus_dataset/releases/download/v1.0"
AISTPP_ANNOTATIONS = ["keypoints3d", "cameras", "splits"]  # keypoints2d (1.3 GB) and motions (SMPL) are not needed
AISTPP_VIDEOS = "https://aistdancedb.ongaaccel.jp/v1.0.0/video/10M"


def _content_length(url: str) -> Optional[int]:
    try:
        request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pedrec-dataset-downloader"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.headers.get("Content-Length") or 0) or None
    except Exception:
        return None


def get_aistpp(datasets: str, accept_terms: bool, views: List[str], max_sequences: Optional[int], workers: int,
               keep_archives: bool, annotations_only: bool = False):
    root = os.path.join(datasets, "AIST++")
    annotations = os.path.join(root, "annotations")
    # annotations: keypoints3d.zip 876 MB, cameras.zip / splits.zip / ignore_list.txt a few KB
    for name in AISTPP_ANNOTATIONS:
        download_and_extract(f"{AISTPP_RELEASE}/{name}.zip", os.path.join(annotations, name), "", keep_archives)
    download(f"{AISTPP_RELEASE}/ignore_list.txt", os.path.join(annotations, "ignore_list.txt"))
    if annotations_only:
        log("annotations done (without videos the converter writes only the lifter sequences)")
        return
    if not accept_terms:
        raise SystemExit("The AIST++ videos are part of the AIST Dance Video Database, read the terms of use "
                         "(https://aistdancedb.ongaaccel.jp/terms_of_use/) and pass --accept-terms "
                         "(or --annotations-only).")
    ignore = set(open(os.path.join(annotations, "ignore_list.txt")).read().split())
    sequences = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(annotations, "keypoints3d"))
                       if f.endswith(".pkl"))
    sequences = [s for s in sequences if s not in ignore][:max_sequences]
    # sequence gBR_sBM_cAll_d04_mBR0_ch01 -> videos gBR_sBM_c01_d04_mBR0_ch01 ... c09
    names = [s.replace("cAll", view) for s in sequences for view in views]
    video_dir = os.path.join(root, "videos")
    missing = [n for n in names if not os.path.isfile(os.path.join(video_dir, f"{n}.mp4"))]
    size = _content_length(f"{AISTPP_VIDEOS}/{missing[0]}.mp4") if missing else None
    estimate = f", ~{size * len(missing) / 1e9:.0f} GB (estimated from the first video)" if size else ""
    log(f"{len(names)} videos ({len(sequences)} sequences x {len(views)} views), {len(missing)} to download"
        f"{estimate} -> {video_dir}")
    from tqdm import tqdm
    failed = []

    def get(name):
        try:
            download(f"{AISTPP_VIDEOS}/{name}.mp4", os.path.join(video_dir, f"{name}.mp4"), quiet=True)
        except Exception as e:
            failed.append(f"{name}: {e}")
    with ThreadPoolExecutor(workers) as pool:
        list(tqdm(pool.map(get, missing), total=len(missing), desc="AIST++ videos", unit="video", dynamic_ncols=True))
    if failed:
        with open(os.path.join(root, "failed_videos.txt"), "w") as f:
            f.write("\n".join(failed) + "\n")
        log(f"{len(failed)} videos failed (see {os.path.join(root, 'failed_videos.txt')}), e.g. {failed[0]}; "
            f"run the task again to retry")
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
    p.add_argument("--workers", type=int, default=None, help="Parallel video decoders (default: CPUs, max 8).")
    p = sub.add_parser("3dhp", help="MPI-INF-3DHP via the official download scripts")
    p.add_argument("--accept-license", action="store_true")
    p.add_argument("--subjects", nargs="*", type=int, default=list(range(1, 9)))
    p.add_argument("--cameras", nargs="*", type=int, default=[0, 1, 2, 4, 5, 6, 7, 8],
                   help="Chest height cameras by default.")
    p = sub.add_parser("aistpp", help="AIST++ annotations + AIST Dance DB videos")
    p.add_argument("--accept-terms", action="store_true")
    p.add_argument("--annotations-only", action="store_true", help="Only the annotations (~0.9 GB), no videos.")
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
        extract_h36m_images(datasets, args.splits, workers=args.workers)
    elif args.command == "3dhp":
        get_3dhp(datasets, args.accept_license, args.subjects, args.cameras, args.keep_archives)
    elif args.command == "aistpp":
        get_aistpp(datasets, args.accept_terms, args.views, args.max_sequences, args.workers, args.keep_archives,
                   args.annotations_only)


if __name__ == "__main__":
    main()
