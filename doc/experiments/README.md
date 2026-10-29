# PedRecNet experiments: comparison values

All validation results of PedRecNet v1 / v2 that are compared against each other, in
[results.csv](results.csv) (one value per row). The tables below summarize them; the CSV is the source.

## Files

| File | Content |
| --- | --- |
| [results.csv](results.csv) | `run, model, stage, checkpoint, dataset, metric, value, note` |
| [../../pedrec/tools/experiments/collect_results.py](../../pedrec/tools/experiments/collect_results.py) | imports training protocols, prints comparison tables |

Columns:

* **run**: `v1-diss` (dissertation) or `v2-<start date of the training>`; several stages belong to one run
* **stage**: training stage (`p2d_c`, `p2d3d_c_o`, v1: `p2d3d_c_o_h36m_sim[_mebow]`)
* **checkpoint**: `R<round>E<epoch>` as in the training log (1-based, `(last)` = last epoch of the protocol), `eval` =
  separate evaluation script (`mise run eval:*`) instead of the validation during training
* **note**: what limits the comparability of this run (read it before comparing)

## Adding results

```bash
# last epoch of a finished stage (or --epochs all / --epochs R2E28,R2E40)
python pedrec/tools/experiments/collect_results.py add data/models/pedrec/single_results/experiment_pedrec_v2_p2d_c_protocol.md \
    --run v2-2026-10-28 --model v2 --stage p2d_c --note "..."
# comparison table
python pedrec/tools/experiments/collect_results.py compare --metrics "COCO PCK@0.05" "H36M MPJPE" --runs v1-diss v2-2026-10-28
```

Rows with the same run / stage / checkpoint / dataset / metric are replaced, so a run in progress (values taken from
the log) is overwritten by its protocol once it is finished.

## Metrics

| Metric | Unit | Meaning |
| --- | --- | --- |
| PCK@0.05 / PCK@0.2 | % | 2D joints within 5 % / 20 % of the person bounding box (GT boxes) |
| MPJPE | mm | mean 3D joint error, root (hip center) relative, all supported joints incl. hand / foot ends |
| PCJP / PCD | fraction | correct relative joint positions / correct relative depth of joint pairs (protocol: MRCJP / MRCD) |
| JointAcc | fraction | joint visibility (confidence) accuracy |
| Body / Head phi deg | ° | mean azimuth error of the orientation (protocols store radians with 2 decimals: ~0.6° resolution) |
| Body / Head sph | - | mean spherical distance of the orientation |
| AP (COCO-eval) | % | COCO keypoint AP (`eval:coco`), the standard 2D measure |
| MPJPE (H36M-eval) | mm | per action mean on the Human3.6m validation subjects (`eval:h36m`), the v1 thesis number |
| Body phi mean deg / Acc22.5 (MEBOW-, TUD-eval) | ° / % | body orientation on COCO-MEBOW / TUD (`eval:coco-orientation`, `eval:tud-orientation`) |

## Comparability

* **SIM validation**: up to `v2-2026-10-23` (and in v1) `rt_validate_3d.pkl` was trained *and* validated, SIM values
  of these runs are training set results. Since `v2-2026-10-28` 4 SIM avatars are held out (`SIM_VAL_HOLDOUT_SUBJECTS`):
  SIM values are only comparable from that run on.
* **Joint conventions**: since `v2-2026-10-28` H36M / MPI-INF-3DHP / Fit3D / AIST++ / COCO are validated in their own
  joint convention (convention adapter), before against the canonical prediction. This mainly affects MPJPE.
* **Fit3D in `v2-2026-10-23`**: left / right legs mirrored in the labels (training and validation).
* **Training validation vs. eval**: the validation during training uses subsampled sets (H36M every 64th frame) and GT
  boxes; `eval` rows come from the evaluation scripts. Compare like with like (v1 H36M: 64.4 mm validation vs. 55.0 mm
  eval for the same network).
* **v1 reference network**: `p2d3d_c_o_h36m_sim_mebow` (demo, SIM-C01 results of the lifter / action recognition).

## Summary (state 2026-10-29)

`v2-2026-10-28` is in progress: p2d_c from the log at round 2 epoch 43 of 60, p2d3d_c_o at round 2 epoch 3 of 40.

### 2D (training validation)

| Metric | v1 p2d_c (COCO only) | v1 reference (mebow) | v2-10-23 p2d_c | v2-10-23 p2d3d_c_o R2E28 | v2-10-28 p2d_c R2E43 | v2-10-28 p2d3d_c_o R2E3 |
| --- | --- | --- | --- | --- | --- | --- |
| COCO PCK@0.05 | 56.44 | 53.72 | 55.11 | 54.65 | 56.21 | 56.11 |
| COCO PCK@0.2 | 92.68 | 91.88 | 92.15 | 92.05 | 92.58 | 92.81 |
| H36M PCK@0.05 | 17.64 | 61.23 | 65.30 | 64.63 | 64.13 | 63.40 |
| H36M PCK@0.2 | 53.17 | 92.66 | 93.72 | 93.69 | 93.66 | 93.61 |
| Fit3D PCK@0.05 | - | - | 43.32 ¹ | 43.63 ¹ | 53.68 | 54.22 |
| AIST++ PCK@0.05 | - | - | 72.93 | 72.36 | 72.58 | 72.66 |
| SIM PCK@0.05 | (30.15) ² | 97.12 ² | 99.35 ² | 98.16 ² | 67.69 | 67.32 |

### 3D and orientation (training validation)

| Metric | v1 reference (mebow) | v1 without MEBOW | v2-10-23 p2d3d_c_o R2E28 | v2-10-28 p2d3d_c_o R2E3 |
| --- | --- | --- | --- | --- |
| H36M MPJPE mm | 64.37 | 64.50 | 87.74 | 79.9 ³ |
| H36M PCJP / PCD | 0.80 / 0.92 | 0.79 / 0.92 | 0.73 / 0.90 | 0.75 / 0.90 |
| Fit3D MPJPE mm | - | - | 65.38 ¹ | 63.4 ³ |
| AIST++ MPJPE mm | - | - | 82.73 | 73.9 ³ |
| SIM MPJPE mm | 15.13 ² | 17.69 ² | 56.42 ² | 74.1 |
| COCO body phi ° (MEBOW) | 10.3 | 16.0 | 9.7 | 10.2 |
| SIM body / head phi ° | 2.3 / 3.4 ² | 4.6 / 3.4 ² | 3.4 / 4.0 ² | 6.5 / 7.8 |

### Evaluation scripts (`mise run eval:*`)

| Metric | v1 p2d_c | v1 without MEBOW | v1 reference (mebow) | v2 |
| --- | --- | --- | --- | --- |
| COCO AP | 70.4 | 67.1 | 67.6 | not evaluated yet |
| H36M MPJPE mm | - | 55.10 | 54.95 | not evaluated yet |
| MEBOW body phi mean ° / Acc22.5 % | - | 16.0 / 80.5 | 10.1 / 91.7 | not evaluated yet |
| TUD body phi mean ° | - | 15.9 | 14.4 | not evaluated yet |

¹ Fit3D legs mirrored. ² SIM validation set was part of the training. ³ validated in the dataset's joint convention.
