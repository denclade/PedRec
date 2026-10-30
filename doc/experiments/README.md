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
* **Canonical vs. adapted 3D (`v2-2026-10-28`)**: the training validation of this run maps the prediction to each
  dataset's convention with the convention adapter. Its root row took 86 % of the hip mean (free, the canonical root
  is the hip mean), so the canonical skeleton drifted ~20 cm from its root on studio images: H36M 74.8 mm adapted vs.
  236.3 mm canonical (`eval:h36m`). The 3D values of this run describe the adapted, not the deployed network. Since the
  fix (root adapts only along the spine) the training also logs the canonical results ("Canonical (no adapter)") and
  selects the best epoch on them.
* **Joints in MPJPE**: the training validation averages all 21 joints with H36M labels incl. hand / foot ends (v1 too),
  `eval:h36m` the standard 17 joints. `v2-2026-10-28` R2E39 adapted: 74.8 mm (21) / 66.2 mm (17).
* **v1 reference network**: `p2d3d_c_o_h36m_sim_mebow` (demo, SIM-C01 results of the lifter / action recognition).

## Summary (state 2026-10-29)

`v2-2026-10-28` is finished (p2d_c R2E60, p2d3d_c_o R2E40). The best checkpoint of p2d3d_c_o is R2E39
(`experiment_pedrec_v2_p2d3d_c_o_0_best.pth`), R2E40 differs by at most 0.1 in every metric. H36M MPJPE is flat from
~R2E25 on (74.6-74.9 mm), SIM / Fit3D / AIST++ still improve slightly until the end.

### 2D (training validation)

| Metric | v1 p2d_c (COCO only) | v1 reference (mebow) | v2-10-23 p2d_c | v2-10-23 p2d3d_c_o R2E28 | v2-10-28 p2d_c R2E60 | v2-10-28 p2d3d_c_o R2E39 |
| --- | --- | --- | --- | --- | --- | --- |
| COCO PCK@0.05 | 56.44 | 53.72 | 55.11 | 54.65 | 56.82 | 55.70 |
| COCO PCK@0.2 | 92.68 | 91.88 | 92.15 | 92.05 | 92.90 | 92.46 |
| H36M PCK@0.05 | 17.64 | 61.23 | 65.30 | 64.63 | 63.71 | 63.28 |
| H36M PCK@0.2 | 53.17 | 92.66 | 93.72 | 93.69 | 93.64 | 93.60 |
| Fit3D PCK@0.05 | - | - | 43.32 ¹ | 43.63 ¹ | 54.04 | 55.00 |
| AIST++ PCK@0.05 | - | - | 72.93 | 72.36 | 72.94 | 72.28 |
| SIM PCK@0.05 | (30.15) ² | 97.12 ² | 99.35 ² | 98.16 ² | 67.61 | 66.90 |

### 3D and orientation (training validation)

| Metric | v1 reference (mebow) | v1 without MEBOW | v2-10-23 p2d3d_c_o R2E28 | v2-10-28 p2d3d_c_o R2E39 |
| --- | --- | --- | --- | --- |
| H36M MPJPE mm | 64.37 | 64.50 | 87.74 | 74.78 ³ |
| H36M PCJP / PCD | 0.80 / 0.92 | 0.79 / 0.92 | 0.73 / 0.90 | 0.77 / 0.91 |
| Fit3D MPJPE mm | - | - | 65.38 ¹ | 55.38 ³ |
| AIST++ MPJPE mm | - | - | 82.73 | 68.25 ³ |
| SIM MPJPE mm | 15.13 ² | 17.69 ² | 56.42 ² | 65.73 |
| COCO body phi ° (MEBOW) | 10.3 | 16.0 | 9.7 | 9.2 |
| SIM body / head phi ° | 2.3 / 3.4 ² | 4.6 / 3.4 ² | 3.4 / 4.0 ² | 4.6 / 4.6 |

### Evaluation scripts (`mise run eval:*`)

| Metric | v1 p2d_c | v1 without MEBOW | v1 reference (mebow) | v2 |
| --- | --- | --- | --- | --- |
| COCO AP | 70.4 | 67.1 | 67.6 | not evaluated yet |
| H36M MPJPE mm | - | 55.10 | 54.95 | 236.3 (`v2-2026-10-28`, canonical, see above) |
| MEBOW body phi mean ° / Acc22.5 % | - | 16.0 / 80.5 | 10.1 / 91.7 | not evaluated yet |
| TUD body phi mean ° | - | 15.9 | 14.4 | not evaluated yet |

¹ Fit3D legs mirrored. ² SIM validation set was part of the training. ³ validated in the dataset's joint convention.
