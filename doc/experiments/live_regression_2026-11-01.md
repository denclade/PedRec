# Live regression: tracking and action recognition

Video: `data/demo/05070850_9672.m4v`, approximately 29.97 fps, 1920 x 1080.
The reported live run used the temporal lifter. Symptoms: overlapping skeletons / ID changes near frames
570–580 and missed standing recognition around frames 500–580.

## Controlled action comparison

CPU inference with the locally available trained weights, frames 430–590 inclusive. A fixed person detection
`[832, 610, 65, 100, 0.99, 0]` (center x/y, width/height, confidence, person class) removes detector variation.
One uninterrupted track, normal One Euro smoothing, 64-frame EHPI input, action threshold 0.7. Frames 430–499
provide history before evaluating the 81 frames 500–580 inclusive.

| Action classifier | v2 pose input | Frames with STAND probability >= 0.7 | Mean STAND probability |
| --- | --- | ---: | ---: |
| Current ST-GCN | without lifter | 75 / 81 | 0.9034 |
| Current ST-GCN | with lifter | 45 / 81 | 0.6110 |
| Published v1 ResNet-50 | without lifter | 79 / 81 | 0.9772 |
| Published v1 ResNet-50 | with lifter | 69 / 81 | 0.8874 |

The v1 classifier received the exact same EHPI images as ST-GCN, produced from the saved smoothed v2 poses.
Its checkpoint uses a ResNet-50 headless feature extractor followed by adaptive average pooling and a linear
classifier, equivalent to torchvision ResNet-50 after removing the `feature_extractor.` state-dict prefix.
This comparison uses `ehpi_3d_sim_c01_actionrec_gt_pred_64frames.pth` versus
`ehpi_stgcn_sim_c01_actionrec_gt_pred_64frames.pth`, with
`experiment_pedrec_v2_p2d3d_c_o_0_net.pth` and optionally `pedrec_v2_lifter.pth`.

This is a diagnostic on one manually selected crop, not a full v1/v2 pipeline benchmark or an annotated accuracy
measurement. It shows that the action failure can occur without an ID change, that the lifter worsens standing
recognition in this clip, and that the published v1 action checkpoint handles these inputs better than the current
ST-GCN checkpoint. It does not isolate architecture from training, or establish general performance on real video.
Even without the lifter, both classifiers miss standing at frame 580.

## Tracking corrections

Before association, suppress near-identical person detections at IoU > 0.8, keeping the highest-confidence box
and original detection indices. This prevents a duplicate box from creating another ID over the same person.
Distinct overlapping boxes below this threshold remain eligible for separate tracks. Very tightly overlapping
people can still be suppressed by this geometric rule; no appearance-based identity model is used.

Gate invalid associations before Hungarian optimization. Previously, globally cheap but invalid pairs could
displace a valid match; discarding those pairs afterwards lost the valid association and could start new IDs.

Regression tests cover duplicate boxes over consecutive frames, payload/index alignment, distinct overlapping
persons and the assignment failure. The exact reported live ID change has not been reproduced: RT-DETR weights
are not cached in this CPU workspace. The actual detector/tracker trajectory remains to be checked on the host.

For the next live comparison, use the corrected tracker with `--no-lifter`. Do not lower action thresholds or
force a STAND fallback to conceal the observed model regression.
