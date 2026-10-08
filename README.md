# PedRecNet
This repository contains the code for the PedRecNet (Paper: https://arxiv.org/pdf/2204.11548.pdf) as well as EHPI3D. It is the successor of our EHPI2D work (https://github.com/noboevbo/ehpi_action_recognition). The PedRecNet is a multi-purpose network that provides the following functions:

- Human BB Detection (via RT-DETRv2).
- Human Tracking
- 2D Human Pose Estimation
- 3D Human Pose Estimation
- Human Body Orientation (currently only Phi) Estimation
- Human Head Orientation (currently only Phi) Estimation
- "Pedestrian recognizes the camera" estimation
- Human Action Recognition (ST-GCN on EHPI skeleton sequences)

Note: This work is currently unpublished. It is part of my PhD dissertation and we are currently in the process to prepare (a? maybe more) paper. Note also, that, for now, I am no longer active in research, thus this code is provided as is.

[![PedRecNet: Demo01 - Pedestrian crossing the street + Hitchhike](https://img.youtube.com/vi/IPeJK1Bk5qY/0.jpg)](https://www.youtube.com/watch?v=IPeJK1Bk5qY)

[![PedRecNet Demo 02: Multiple Pedestrians](https://img.youtube.com/vi/xUcTKKGHfEs/0.jpg)](https://www.youtube.com/watch?v=xUcTKKGHfEs)

# PedRec v2 (branch `v2`)
`v2` is the reworked network and pipeline (one configuration, no switches for the old components). `main` keeps the
published v1.

| Part | v1 (`main`) | v2 |
| --- | --- | --- |
| Detector | YoloV4 608 x 320 + NMS (33.8 GMACs) | RT-DETRv2 R18, NMS free, aspect preserving 640 x 352 for 16:9 (~28 GMACs) |
| PedRecNet backbone | ResNet-50 + three deconvolutions (6.28 GMACs / crop) | HGNetV2-B3 + FPN style neck (2.45 GMACs / crop), same 64 x 48 heatmaps |
| Orientation | regression of theta / phi | biternion (cos, sin) with cosine loss |
| Joint confidence | FC head | heatmap statistics (peak, entropy, spread) + BCE with logits |
| Coordinates | original affine transform | UDP (unbiased data processing) |
| Multi task loss | learned sigmas | Kendall log variances, clamped |
| Training augmentation | flip, scale, rotation | + half body, color jitter, random erasing; optional dataset balancing |
| 3D pose over time | 2 frame mean | causal temporal lifter per track (27 frames, 0.7M parameters) |
| Tracking | optical flow + pose merging | ByteTrack + One Euro filter (orientations) |
| Action recognition | ResNet-50 on the EHPI image (167 MMACs) | ST-GCN on the same EHPI input (116 MMACs, 0.2M parameters) |

Pipeline timings on the CPU (4 threads, random weights, 1920 x 1080, 6 persons, median ms per frame, `mise run bench
--random-weights --cpu`; on the GPU run `mise run bench --fast` on both branches):

| Stage | v1 | v2 |
| --- | --- | --- |
| detection | 509 | 362 |
| pose (6 persons) | 544 | 249 |
| 3D lifting | - | 5 |
| action recognition | 51 | 30 |
| total | 1129 | 648 |

The data formats are the same (dataframes, heatmaps, EHPI images), so all datasets are used unchanged. The v1 weights
do not fit v2; v2 is trained with this code (backbone from ImageNet):

```bash
mise run download:models   # RT-DETR detector + ImageNet weights of the backbone (Hugging Face cache)
mise run train:all         # PedRecNet stages -> demo weights -> 3D lifter -> action recognition
```

# Citation
Please cite the following paper if this code is helpful in your research.

```bash
D. Burgermeister and C. Curio, “PedRecNet: Multi-task deep neural network for full 3D human pose and orientation estimation,” in 2022 IEEE Intelligent 
Vehicles Symposium (IV), 2022.
```

# Quick start
The repository uses [mise](https://mise.jdx.dev) as task runner and environment manager. All entry points (demo, training,
evaluation, tools) are available as `mise` tasks (`mise tasks` lists them) and as plain Python scripts with `--help`.

```bash
mise install                 # pinned Python 3.14 + uv
mise run setup               # .venv with PyTorch 2.14 (CUDA 13.0), PyQt6, ...; see below for other CUDA versions
mise run gpu:info            # check that PyTorch sees the GPU
mise run download:models     # RT-DETR detector + backbone ImageNet weights (Hugging Face cache)
mise run download:datasets   # dataset overview, e.g. download:datasets:pedrec / :coco (see Datasets)
mise run data:check           # what is downloaded / converted / missing, next steps (run before training)
mise run train:all           # train the v2 weights (see above), then:
mise run demo --video my_video.mp4
mise run test                # unit tests (no data / GPU needed)
```

### GPU / CUDA
The code runs on current consumer GPUs incl. the RTX 50xx (Blackwell, e.g. RTX 5070 / 5080, compute capability 12.0):

| Task | PyTorch wheels | NVIDIA driver |
| --- | --- | --- |
| `mise run setup` | PyPI (Linux): CUDA 13.0, sm_75 - sm_120 | >= 580 |
| `mise run setup:cu128` | CUDA 12.8 (also the GPU build for Windows) | >= 570 |
| `mise run setup:cpu` | CPU only | - |

Training uses bf16 mixed precision by default (`--amp`), the default batch size of 48 fits into 12 GB; with less memory
use e.g. `--batch-size 24 --accumulate 2` (same effective batch size).

Data (datasets, models, demo videos) live below the data root, `./data` by default. Change it via the
`PEDREC_DATA_DIR` environment variable, e.g. in a git-ignored `mise.local.toml`:

```toml
[env]
PEDREC_DATA_DIR = "/mnt/storage/pedrec_data"
```

## Manual installation (without mise)
- Python 3.12 or newer (3.14 tested), venv suggested
- `pip install -r requirements.txt` (`requirements-dev.txt` for the tests)
- Run the scripts from the repository root, e.g. `python pedrec/demo.py --help`

## Compatibility with the original data and weights
All datasets / dataframes work unchanged. The dataframes were pickled with pandas 1.3 / numpy 1.21; they are read through
`pedrec.utils.pandas_helper.read_pedrec_df`, which converts numeric categorical columns that current pandas can not
sum (tested with fixtures written by the original versions, `tests/data`). Optionally rewrite them once with the
current versions: `mise run tools:convert-dfs data/datasets/ROMb/rt_rom_01b.pkl ...` (keeps a `.bak` copy).

## Required Data
### Pretrained models
`mise run download:models` fetches the RT-DETRv2 detector (Apache 2.0, `PekingU/rtdetr_v2_r18vd`, `--rtdetr-model`
selects another size or a local copy) and the ImageNet weights of the PedRecNet backbone (timm `hgnetv2_b3`) into the
Hugging Face cache. The v2 weights are trained with this code (`mise run train:all`) and expected in

- *data/models/pedrec/experiment_pedrec_v2_p2d3d_c_o_0_net.pth* (PedRecNet, `mise run tools:extract-net --stage p2d3d_c_o`)
- *data/models/pedrec/pedrec_v2_lifter.pth* (temporal 3D lifter)
- *data/models/ehpi3d/ehpi_stgcn_sim_c01_actionrec_gt_pred_64frames.pth* (action recognition)

### Datasets
All datasets are expected below `$PEDREC_DATA_DIR/datasets` (default *data/datasets*). `mise run download:datasets`
prints an overview, the downloads are resumable and skip existing files. `mise run data:check` shows what is downloaded,
converted and missing (incl. sampled images at the dataframe rows the training reads), whether `train:pedrec` /
`train:lifter` / `train:all` can run, the commands for the next steps and the recommended additional datasets. Licenses: all datasets are restricted to
non-commercial research (COCO annotations: CC BY 4.0), check them before use.

| Dataset | Content | Used for | Get it |
|---|---|---|---|
| COCO 2017 | 2D, in the wild | PedRecNet 2D / confidence | `mise run download:datasets:coco` |
| MEBOW | COCO body orientations | PedRecNet orientation | by e-mail, see `mise run download:datasets:info mebow` |
| Human3.6m | 3D studio (50 fps) | PedRecNet 3D, lifter | registration (`info h36m`), then `download:datasets:pedrec` (dataframes) + `data:h36m:images` |
| SIM-ROM (ROMb), SIM-Circle (RT3DValidate) | simulated pedestrians, range of motion, orientations | PedRecNet 3D / orientation, lifter | `mise run download:datasets:pedrec` |
| SIM-C01 | simulated pedestrian actions (dataframes only) | action recognition, lifter | `mise run download:datasets:pedrec` |
| TUD multiview pedestrians | orientation benchmark | evaluation | `mise run download:datasets:info tud` |
| **MPI-INF-3DHP** (new) | 3D, 8 actors, green screen + outdoor test set, 14 cameras | PedRecNet 3D (images), lifter | `mise run download:datasets:3dhp --accept-license`, `mise run data:convert:3dhp` |
| **Fit3D** (new) | fitness exercises with large range of motion, 4 cameras; also HumanSC3D / CHI3D | PedRecNet 3D (images), lifter | registration (`info fit3d`), `mise run data:convert:fit3d` |
| **AIST++** (new) | dance (jumps, spins, floor moves), 9 views, 10M frames | PedRecNet 3D (images), lifter | `mise run download:datasets:aistpp --accept-terms [--views c01 c05 --max-sequences 300]` (annotations 0.9 GB, videos: size printed before the download, a few hundred GB for all views; `--annotations-only` for the lifter only), `mise run data:convert:aistpp` |
| **AMASS** (new) | 40+ hours of mocap as SMPL-H, incl. **PosePrior** (formerly MPI_Limits: joint limits / range of motion) | lifter only (no images) | registration (`info amass`), `mise run data:convert:amass` |

The converters (`pedrec/tools/datasets/convert_*.py`) write PedRec dataframes (`<name>_{train,val}_pedrec.pkl`,
images `img_<frame>.jpg`, same columns as the H36M / SIM dataframes: 26 joints 2D + 3D in mm relative to the hip,
`supported` = 0 for joints the dataset does not have) and sequence dataframes for the lifter
(`<name>_{train,val}_seq.pkl`). The conversion decodes the videos in parallel (`--workers`, default: CPUs, max 8)
with a progress bar and can be interrupted and restarted (extracted images are kept). MPI-INF-3DHP images are
downscaled from 2048x2048 to 1024x1024 (`--max-image-size`, 0 = original size; the crops of PedRecNet are 192x256
anyway). Expect hours for the full MPI-INF-3DHP / AIST++ conversion (all frames of all videos have to be decoded). As soon as they exist, `train:pedrec` uses them for training and validation
(disable with `--no-extra-3d`, balance with e.g. `--dataset-weights coco=1,h36m=1,sim=1,mpi_inf_3dhp=0.5,fit3d=0.5,aistpp=0.5`)
and `train:lifter` adds the sequences. The new datasets have no orientation labels (orientation loss masked).

Recommendations: AMASS (at least PosePrior + CMU) gives the temporal lifter a far larger pose / motion variety than
H36M + SIM; Fit3D and AIST++ add real images of extreme poses (range of motion) for the 3D head of PedRecNet,
MPI-INF-3DHP adds outdoor / unusual camera views. BEDLAM (synthetic, many people, 2D + 3D, registration) and
AthletePose3D (sports, CVPR 2025) are further candidates but not integrated.

The expected layout below the data root (see `pedrec/training/experiments/experiment_path_helper.py`, every path can be
overridden there or via the script options):

```
data/
  datasets/COCO, Human3.6m/{train,val}, ROMb, RT3DValidate, cvpr10_multiview_pedestrians, Conti01,
           MPI-INF-3DHP, Fit3D, AIST++, AMASS (optional)
  models/pedrec, models/pedrec/single_results (training checkpoints), models/ehpi3d, models/body_models/smplh (AMASS)
  demo/
```

### Demo files
- [Some C01 real examples](https://dennisnotes.com/files/pedrec/demo/05070850_9672.m4v) - `mise run download:demo` or place it in *data/demo/*.
- [Pedestrians crossing a street](https://www.pexels.com/de-de/video/855565/) - place it as *data/demo/multi_person_crossing_street.mp4* (the default demo input).

## Installation tips
Currently I would recommend to use a PIP environment instead of Anaconda. I tried the (recommended) Anaconda environment for PyTorch various times, but the performance is hugely inferior to the PIP environment on my system(s). Using Anaconda I get about 9FPS on videos with a single human compared to 25FPS on my PIP environment. One thing I noticed is that the performance difference shrinks the more people are in a video, thus with 7+ people the performance of the Anaconda and the PIP environment are almost equal. If someone has an idea what the problem could be, please notify me. Things tested:

- CUDA / CUDNN are working enabled and recognized by PyTorch on both environments
- Pillow-SIMD installed
- Usage of opencv-contrib-python-headless instead of the Conda version.

# Demo / Run
`pedrec/demo.py` runs the pipeline (RT-DETR -> PedRecNet -> ByteTrack + One Euro filter -> ST-GCN) on videos, image
directories, single images or a webcam, with the Qt GUI or headless. Every stage can be switched off, so the parts of
the network can be run on their own.

| Task | What it does |
| --- | --- |
| `mise run demo` | Qt GUI on the default demo video (`--video`, `--images`, `--image`, `--webcam` select the input) |
| `mise run demo:video <file>` / `demo:images <dir>` / `demo:image <file>` / `demo:webcam [id]` | Qt GUI on the given input |
| `mise run demo:fast` | GUI with fp16 autocast + channels_last (Tensor Cores) |
| `mise run demo:headless --video in.mp4 --output out.mp4 --json out.json` | Full pipeline without GUI, writes an annotated video / images and all results as JSON |
| `mise run demo:detector --video in.mp4 --output out.mp4` | RT-DETR detector only |
| `mise run demo:pose --image person.jpg --output out.jpg --json out.json` | PedRecNet only: 2D / 3D pose + orientation on the full frame (no detector, tracking, actions) |
| `mise run demo:detector-pose --video in.mp4 --output out.mp4` | Detector + PedRecNet without tracking / actions |
| `mise run demo:no-action` | GUI without action recognition |
| `mise run bench` | Timings of the networks and of the pipeline per stage (`--fast`, `--video`, `--random-weights`) |

GUI player controls (below the video): play / pause (`Space`), previous / next frame while paused (`Left` /
`Right`, while playing they skip 5 s), replay (`Home`), skip back / forward and a seek bar. Processed frames are cached,
so stepping back, replaying and seeking into the processed part show the same tracked results instantly; after a jump
into an unprocessed part the tracking starts anew. Left of the video a magnifier shows the plain camera image of the
selected person (no overlays) to see what the person is doing; the button in its corner pops it out into an own,
resizable window (closing the window docks it again).

Useful options (see `python pedrec/demo.py --help`):
- stages: `--no-detector`, `--no-pose`, `--no-tracking`, `--no-action`, `--action-list c01|c01_real`
- speed: `--fast` (= `--half --channels-last`), `--compile`, `--prefetch N`, `--cpu`
- `--no-lifter` (per frame 3D poses), `--size WxH`, `--max-frames N`, `--rtdetr-model`, `--pedrec-weights`,
  `--lifter-weights`, `--ehpi3d-weights`

The pipeline (`pedrec/inference/pipeline.py`, `PedRecPipeline`) can be embedded in your own code. It uploads every
frame once, runs the person crops, normalization and coordinate transformations batched on the GPU and runs PedRecNet
exactly once per frame. The batched path is tested against a plain OpenCV implementation
(`tests/test_inference_equivalence.py`).

# Training
`mise run train:all` runs the complete chain; the single steps:

```bash
mise run train:list                       # stage table
mise run train:pedrec:2d                  # stage p2d_c: 2D pose + joint confidence, backbone from ImageNet
mise run train:pedrec:full                # stage p2d3d_c_o: + 3D pose + orientation (demo network)
mise run tools:extract-net --stage p2d3d_c_o
mise run train:lifter:data                # PedRecNet results on SIM-C01 (input of the lifter training)
mise run train:lifter                     # temporal 3D lifter
mise run train:ehpi3d:data                # SIM-C01 results with the lifted 3D poses
mise run train:ehpi3d                     # action recognition (ST-GCN)
```

## PedRecNet
Two stages (`pedrec/training/experiments/pedrec_stages.py`): `p2d_c` trains the neck and the 2D / confidence heads on
COCO, Human3.6m and SIM starting from the ImageNet backbone, `p2d3d_c_o` adds the 3D pose and the orientations (COCO
with the MEBOW labels). Each stage trains two rounds (frozen backbone, then the full network), writes
`experiment_pedrec_v2_<stage>_0_01.pth` / `experiment_pedrec_v2_<stage>_0.pth` (EMA weights), the best epoch
`experiment_pedrec_v2_<stage>_0_best.pth` and a markdown protocol into *data/models/pedrec/single_results/*. The epochs
per round are stage defaults (`--epochs-round-1/2` override them). Without `--lr` the LR range test runs first (plot
next to the checkpoint). The schedules are starting points: check the protocols of the first runs.

Training stability options (PedRecNet and action recognition):

| Option | Default | Effect |
| --- | --- | --- |
| `--amp auto/bf16/fp16/off` | auto (bf16 on RTX 30xx-50xx) | mixed precision; loss heads always in fp32, `off` = original fp32 training |
| `--grad-clip N` | 10 | clip the global gradient norm |
| `--accumulate N` | 1 | gradient accumulation for GPUs with less memory |
| `--ema-decay D` | 0.9998 (action recognition 0.999) | exponential moving average of the weights, used for validation and the checkpoints |
| `--resume` | - | continue an interrupted stage from `experiment_pedrec_v2_<stage>_<cycle>_state.pth` (written every epoch) |

Steps with a non-finite loss or gradient are skipped (the run stops after 50 in a row). The best epoch is selected by
the mean relative improvement of all validation metrics (PCK, MPJPE, joint accuracy, orientation errors) compared to
the first epoch. `--dataset-weights coco=1,h36m=1,sim=1` samples the training datasets with the given
relative probabilities instead of proportional to their size.

## Temporal 3D lifter
`pedrec/training/train_lifter.py` trains the causal lifter on 3D ground truth sequences of Human3.6m, SIM-ROM and
SIM-C01 (only the dataframes are needed). Inputs are the PedRecNet predictions where available (SIM-C01 results of
`train:lifter:data`) and otherwise the ground truth with noise that imitates the per frame errors. Every epoch reports
the 3D error (MPJPE) of the per frame PedRecNet poses and of the lifted poses per validation set; the best epoch is
saved, and a warning is logged if the lifter does not improve on the per frame poses.

## Action recognition (ST-GCN on EHPI sequences)
```bash
mise run train:ehpi3d:list                     # variants (gt / pred / mixed skeletons x 30 fps / 15 fps / 64 frames)
mise run train:ehpi3d --variant gt_pred_64frames
```

The training uses the SIM-C01 skeleton dataframes and the PedRecNet results on them with the lifted 3D poses
(`*_allframes_lifted.pkl`, `mise run train:ehpi3d:data`), i.e. the same 3D poses the pipeline produces at runtime.
`mise run tools:ehpi-videos` creates the skeleton dataframe of the real EHPI videos with the full pipeline.

# Evaluation
| Task | What it does |
| --- | --- |
| `mise run eval:pedrec --weights <ckpt>` | 2D / 3D pose, joint confidence and orientation metrics on COCO, SIM and Human3.6m (training validation code) |
| `mise run eval:h36m --weights <net.pth>` | Per-action MPJPE on Human3.6m |
| `mise run eval:coco-orientation --weights <net.pth>` | Body orientation accuracy on COCO (MEBOW) |
| `mise run eval:tud-orientation --weights <net.pth>` | Body orientation accuracy on TUD |
| `mise run eval:ehpi3d --variants gt_pred_64frames` | Action recognition metrics on SIM-C01 |
| `mise run results:coco` / `results:h36m` / `results:sim-c01` | Write PedRecNet result dataframes (`--experiments <stages>`) |
| `mise run export:coco` / `export:h36m` / `export:sim-c01` | Markdown / LaTeX result tables from the result dataframes (as used in `doc/diss_eval`) |

# Project structure
```
pedrec/demo.py                  demo / inference CLI (GUI + headless)
pedrec/inference/pipeline.py    PedRecPipeline: detector -> pose -> tracking -> 3D lifting -> actions
pedrec/inference/gpu_ops.py     batched device side crops, resize, coordinate transforms
pedrec/tracking/                ByteTrack style tracker, One Euro filter
pedrec/training/train_pedrec.py PedRecNet training (stages: pedrec/training/experiments/pedrec_stages.py)
pedrec/training/train_lifter.py temporal 3D lifter training
pedrec/training/train_ehpi3d.py action recognition training (variants: pedrec/training/experiments/ehpi3d_variants.py)
pedrec/evaluations/             validation / evaluation scripts
pedrec/tools/                   dataset generators, result writers, weight tools
pedrec/networks/                PedRecNet, temporal 3D lifter, ST-GCN, RT-DETR wrapper
pedrec/datasets/, pedrec/configs/, pedrec/utils/, pedrec/tracking/, pedrec/ui/, pedrec/visualizers/
doc/                            experiment protocols, evaluation results, architecture review
tests/                          unit tests (mise run test)
```

# Generate own training data
Check out the panda dataframes (e.g. the rt_conti_01_train_FIN.pkl from SIM-C01 dataset, or the pkls from the H36M dataset). If you provide a dataset of the same structure you can just use the pedrec dataset class.
You can find some scripts I used to generate the dataframes in `pedrec/tools/datasets/`, but I have not tested them in a while.
The same applies for EHPI3D action recognition data: Check out the dataframes from the rt_conti_01_train_FIN.pkl file! You might want to checkout the notebook *dataset_rtsim_conti01_ehpi* as well. The PedRecNet result dataframes for SIM-C01 can be regenerated with `mise run train:lifter:data`. You can find the result files of v1 (e.g. the C01F_train_pred_df_experiment_pedrec_p2d3d_c_o_h36m_sim_mebow_0_allframes.pkl) at https://dennisnotes.com/files/pedrec/result_dfs/filename; v2 writes its own (`experiment_pedrec_v2_p2d3d_c_o_0`).

# Notebooks
I've just pasted a few of my notebooks in the notebooks folder. They are not cleaned up and may contain absolute paths etc. but maybe they help the one or other to understand some concepts / validation results.

# Appendix
Note: probably outdated information! Need to recheck this part.

## Numpy "Datatypes"
note: not really datatypes, those types are stored in numpy arrays due to performance considerations.
There are helper methods providing more userfriendly access to those values (e.g. joint_helper(_3d), bb_helper).
Those datatypes are the ones used internally in the PedRecNet application, there might be differences in types used in e.g. datasets etc.

| "Datatype" name | Shape |
| -------------------------- |----------- |
| bb_2d | center_x, center_y, width, height, confidence, class_idx |
| joint_2d | x, y, confidence | 
| joint_3d | x, y, z, confidence | 

## Expected shapes of PedRecNet HDF5 dataset files
note: n = dataset length

| dataset name               | Shape      | DType   | Description |
| -------------------------- |----------- | ------- | ----------  |
| img_paths                  | (n)        | str     |img path, relative to the dataset root |
| joints2d                   | (n,17,4)   | float32 |17 = joints, 4 = x, y, confidence, visibility (coordinates in pixels, starting from top left of the image) |
| skeleton_3d_hip_normalized | (n,17,5)   | float32 | 17 = joints, 5 = x, y, z, confidence, visibility (coordinates in mm) |
| env_position               | (n,3)      | float32 | 3 = x, y, z (mm) |
| body_orientation           | (n,4)      | float32 | 4 = theta, phi, confidence, visibility |
| head_orientation           | (n,4)      | float32 | 4 = theta, phi, confidence, visibility |
| bbs                        | (n,6)      | float32 | 5 = center_x, center_y, width, height, confidence, class_idx |
| scene_idx_range            | (n,2)      | uint32  | 2 = scene_idx_start, scene_idx_stop the index range in the hdf5 file containing data from the same scene |
| actions                    | (n)        | uint32  | List = dynamic sized list of action ids, e.g. [[1, 2], [3, 4, 5]] |
| movements                  | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| movement_speeds            | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| genders                    | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| skin_colors                | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| sizes                      | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| weights                    | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| ages                       | (n)        | uint32  | ids, see constants for ID <-> NAME mapping |
| frame_nr_locals            | (n)        | uint32  | frame number of the current scene |
| frame_nr_global            | (n)        | uint32  | frame number of the complete record |


## Original dataset notes
Some notes to original datasets.
Important: Those notes do NOT apply to internal PedRec usage, the original datasets are converted to PedRec Datasets before usage, thus those notes can usually be ignored.

### Human3.6M

#### BB Structure
They use a binary mask containing 1s in the bounding box area.
#### Joint Order
- 0 = 'Hips'
- 1 = 'RightUpLeg'
- 2 = 'RightLeg'
- 3 = 'RightFoot'
- 4 = 'RightToeBase'
- 5 = 'Site'  - ????
- 6 = 'LeftUpLeg'
- 7 = 'LeftLeg'
- 8 = 'LeftFoot'
- 9 = 'LeftToeBase'
- 10 = 'Site'  - ????
- 11 = 'Spine'
- 12 = 'Spine1'
- 13 = 'Neck'
- 14 = 'Head'
- 15 = 'Site'
- 16 = 'LShoulder'
- 17 = 'LeftArm'
- 18 = 'LeftForeArm'
- 19 = 'LeftHand'
- 20 = 'LeftHandThumb'
- 21 = 'Site'
- 22 = 'L_Wrist_End'
- 23 = 'Site'
- 24 = 'RightShoulder'
- 25 = 'RightArm'
- 26 = 'RightForeArm'
- 27 = 'RightHand'
- 28 = 'LeftHandThumb'
- 29 = 'Site'
- 30 = 'L_Wrist_End'
- 31 = 'Site'

# Attributions
- RT-DETRv2 object detection: https://github.com/lyuwenyu/RT-DETR (weights via Hugging Face transformers)
- ST-GCN: Yan et al., AAAI 2018; adaptive graph as in 2s-AGCN, Shi et al., CVPR 2019
- Pose-resnet as base network: https://github.com/microsoft/human-pose-estimation.pytorch

## Icons
- Skeleton by Wolf Böse from the Noun Project
- Head by Naveen from the Noun Project
- body by Makarenko Andrey from the Noun Project
- Eye by Simon Sim from the Noun Project
- jogging by Adrien Coquet from the Noun Project
- Walk by Adrien Coquet from the Noun Project
- stand by Gan Khoon Lay from the Noun Project
- sit by Adrien Coquet from the Noun Project

# Contact
- Dennis Burgermeister, Cognitive Systems Research Group, Reutlingen University (no longer active)
- Cristóbal Curio, Cognitive Systems Research Group, Reutlingen University

# Acknowledgment
This project was funded by the Continental AG.
