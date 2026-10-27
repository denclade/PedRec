import logging
import math
from typing import List

from pedrec.models.experiments.experiment_description import ExperimentDescription
from pedrec.models.experiments.epoch_validation_results import EpochValidationResults
from pedrec.utils.string_helper import get_seconds_as_string


def results_to_csv(epoch_validation_results: List[EpochValidationResults]):
    header_fields = get_header_fields(epoch_validation_results)
    output = f"{','.join(header_fields)}\n"
    for epoch_results in epoch_validation_results:
        output += ','.join(get_epoch_values(epoch_results)) + "\n"
    return output


def results_to_md_table(epoch_validation_results: List[EpochValidationResults]):
    header_fields = get_header_fields(epoch_validation_results)
    output = f"| {' | '.join(header_fields)} |\n"
    output += f"|"
    for field in header_fields:
        output += f" {'-' * len(field)} |"
    output += "\n"
    for epoch_results in epoch_validation_results:
        values = get_epoch_values(epoch_results)
        output += f"|"
        for field, value in zip(header_fields, values):
            output += f" {value.ljust(len(field))} |"
        output += "\n"
    return output


# Terminal table columns: (header, value getter, format). Orientation phi errors in degrees, the spherical distance is
# the chord length on the unit sphere (0..2). Only columns with a value in at least one validation set are shown.
_TABLE_COLUMNS = [
    ("PCK@0.05", lambda r: r.pose2d_pck and r.pose2d_pck.pck_05_mean, "{:.2f}"),
    ("PCK@0.2", lambda r: r.pose2d_pck and r.pose2d_pck.pck_2_mean, "{:.2f}"),
    ("MPJPE mm", lambda r: r.pose3d and r.pose3d.mpjpe_mean, "{:.1f}"),
    ("PCJP %", lambda r: r.pose3d and r.pose3d.pct_correct_joint_position_mean, "{:.2f}"),
    ("PCD %", lambda r: r.pose3d and r.pose3d.pct_correct_depth_mean, "{:.2f}"),
    ("JointAcc", lambda r: r.pose2d_conf and r.pose2d_conf.conf_acc, "{:.3f}"),
    ("Body sph", lambda r: r.orientation and r.orientation.body.spherical_distance, "{:.3f}"),
    ("Body phi deg", lambda r: r.orientation and math.degrees(r.orientation.body.angle_error_phi), "{:.1f}"),
    ("Head sph", lambda r: r.orientation and r.orientation.head.spherical_distance, "{:.3f}"),
    ("Head phi deg", lambda r: r.orientation and math.degrees(r.orientation.head.angle_error_phi), "{:.1f}"),
    ("Env mm", lambda r: r.env_position and r.env_position.distance_mm, "{:.1f}"),
    ("Val loss", lambda r: r.loss, "{:.4f}"),
    ("Time", lambda r: get_seconds_as_string(r.val_duration), "{}"),
]


def format_epoch_validation_table(results: EpochValidationResults) -> str:
    """One row per validation set, one column per metric (only the metrics validated on at least one set)."""
    rows = {name: [getter(result) for _, getter, _ in _TABLE_COLUMNS]
            for name, result in results.validation_results.items()}
    used = [i for i in range(len(_TABLE_COLUMNS)) if any(values[i] is not None for values in rows.values())]
    headers = ["Validation set"] + [_TABLE_COLUMNS[i][0] for i in used]
    cells = [[name] + ["-" if values[i] is None else _TABLE_COLUMNS[i][2].format(values[i]) for i in used]
             for name, values in rows.items()]
    widths = [max(len(row[c]) for row in [headers] + cells) for c in range(len(headers))]

    def line(row):
        return "  " + "  ".join(cell.ljust(w) if c == 0 else cell.rjust(w)
                                for c, (cell, w) in enumerate(zip(row, widths)))

    return "\n".join([line(headers), "  " + "-" * (sum(widths) + 2 * (len(widths) - 1))] + [line(row) for row in cells])


def log_epoch_validation_results(results: EpochValidationResults, title: str = None, lr: float = None):
    """Epoch summary for the terminal: title line with train loss / time (/ lr) and the validation table."""
    logger = logging.getLogger(__name__)
    title = title or f"Epoch {results.epoch + 1}"
    summary = f"{title} | train loss {results.train_loss:.4f} | train time {get_seconds_as_string(results.train_time)}"
    if lr is not None:
        summary += f" | max lr {lr:.2e}"
    logger.info(f"{summary}\n{format_epoch_validation_table(results)}")


def get_header_fields(epoch_validation_results: List[EpochValidationResults]) -> List[str]:
    fields = []
    fields.append("Epoch")
    fields.append("Train Loss")
    for val_set_name, result in epoch_validation_results[0].validation_results.items():
        if result.pose2d_pck is not None:
            fields.append(f"{val_set_name} PCK@0.05")
            fields.append(f"{val_set_name} PCK@0.2")
        if result.pose3d is not None:
            fields.append(f"{val_set_name} MPJPE")
            fields.append(f"{val_set_name} MRCJP")
            fields.append(f"{val_set_name} MRCD")
        if result.pose2d_conf is not None:
            fields.append(f"{val_set_name} JointAcc")
        if result.orientation is not None:
            fields.append(f"{val_set_name} Body Sph.Dist.")
            fields.append(f"{val_set_name} Body.Phi.Ang.Dist.")
            fields.append(f"{val_set_name} Head Sph.Dist.")
            fields.append(f"{val_set_name} Head.Phi.Ang.Dist.")
        if result.env_position is not None:
            fields.append(f"{val_set_name} EnvPosDistMM.")
        fields.append(f"{val_set_name} Val Loss")
        fields.append(f"{val_set_name} Val Time")
    fields.append("Train Time")
    return fields


def get_epoch_values(epoch_results: EpochValidationResults) -> List[str]:
    values: List[str] = []
    values.append(f"{epoch_results.epoch}")
    values.append(f"{epoch_results.train_loss:.5f}")
    for val_set_name, result in epoch_results.validation_results.items():
        if result.pose2d_pck is not None:
            values.append(f"{result.pose2d_pck.pck_05_mean:.2f}")
            values.append(f"{result.pose2d_pck.pck_2_mean:.2f}")
        if result.pose3d is not None:
            values.append(f"{result.pose3d.mpjpe_mean:.2f}")
            values.append(f"{result.pose3d.pct_correct_joint_position_mean:.2f}")
            values.append(f"{result.pose3d.pct_correct_depth_mean:.2f}")
        if result.pose2d_conf is not None:
            values.append(f"{result.pose2d_conf.conf_acc:.2f}")
        if result.orientation is not None:
            values.append(f"{result.orientation.body.spherical_distance:.2f}")
            values.append(f"{result.orientation.body.angle_error_phi:.2f}")
            values.append(f"{result.orientation.head.spherical_distance:.2f}")
            values.append(f"{result.orientation.head.angle_error_phi:.2f}")
        if result.env_position is not None:
            values.append(f"{result.env_position.distance_mm:.2f}")
        values.append(f"{result.loss:.4f}")
        values.append(f"{get_seconds_as_string(result.val_duration)}")
    values.append(f"{get_seconds_as_string(epoch_results.train_time)}")
    return values


def get_experiment_protocol(experiment_description: ExperimentDescription):
    md = f"# {experiment_description.net_name}\n" \
         f"## Trial Name\n" \
         f"**{experiment_description.experiment_name}**\n" \
         f"## Initialization\n" \
         f"| {'Network Part'.ljust(30)} | {'Initialization'.ljust(60)} |\n" \
         f"| {'-' * 30} | {'-' * 60} |\n"
    for network_part in experiment_description.net_layer_names:
        md += f"| {network_part.ljust(30)} | {'TODO'.ljust(60)} |\n"

    md += f"Notes. {experiment_description.initialization_notes}\n" \
          f"## Datasets\n" \
          f"### Training\n" \
          f"| {'Dataset'.ljust(25)} | Subsampling | Full set length | Used length | Sampling share |\n" \
          f"| {'-' * 25} | {'-' * 11} | {'-' * 15} | {'-' * 11} | {'-' * 14} |\n"
    total_length = sum(dataset.used_length for dataset in experiment_description._train_sets)
    for dataset in experiment_description._train_sets:
        share = dataset.sampling_share
        if share is None:
            share = dataset.used_length / max(total_length, 1)
        md += f"| {dataset.name.ljust(25)} | {str(dataset.subsampling).ljust(11)} | {str(dataset.full_length).ljust(15)} | {str(dataset.used_length).ljust(11)} | {f'{100 * share:.1f}%'.ljust(14)} |\n"
    if experiment_description.dataset_sampling_weights:
        md += "Dataset weights: " + ", ".join(
            f"{k}={v:g}" for k, v in experiment_description.dataset_sampling_weights.items()) + "\n"
    md += f"### Validation\n" \
          f"| {'Dataset'.ljust(25)} | Subsampling | Full set length | Used length |\n" \
          f"| {'-' * 25} | {'-' * 11} | {'-' * 15} | {'-' * 11} |\n"
    for dataset in experiment_description._val_sets:
        md += f"| {dataset.name.ljust(25)} | {str(dataset.subsampling).ljust(11)} | {str(dataset.full_length).ljust(15)} | {str(dataset.used_length).ljust(11)} |\n"
    md += f"## Augmentation\n" \
          f"### Training Augmentations - COCO\n" \
          f"| {'Augmentation'.ljust(25)} | {'Value'.ljust(25)} |\n" \
          f"| {'-' * 25} | {'-' * 25} |\n" \
          f"| {'Scale'.ljust(25)} | {str(experiment_description.coco_train_dataset_cfg.scale_factor).ljust(25)} |\n" \
          f"| {'Flip'.ljust(25)} | {str(experiment_description.coco_train_dataset_cfg.flip).ljust(25)} |\n" \
          f"| {'Rotate'.ljust(25)} | {str(experiment_description.coco_train_dataset_cfg.rotation_factor).ljust(25)} |\n" \
          f"### Training Augmentations - SIM\n" \
          f"| {'Augmentation'.ljust(25)} | {'Value'.ljust(25)} |\n" \
          f"| {'-' * 25} | {'-' * 25} |\n" \
          f"| {'Scale'.ljust(25)} | {str(experiment_description.sim_train_dataset_cfg.scale_factor).ljust(25)} |\n" \
          f"| {'Flip'.ljust(25)} | {str(experiment_description.sim_train_dataset_cfg.flip).ljust(25)} |\n" \
          f"| {'Rotate'.ljust(25)} | {str(experiment_description.sim_train_dataset_cfg.rotation_factor).ljust(25)} |\n" \
          f"### Training Augmentations - H36M\n" \
          f"| {'Augmentation'.ljust(25)} | {'Value'.ljust(25)} |\n" \
          f"| {'-' * 25} | {'-' * 25} |\n" \
          f"| {'Scale'.ljust(25)} | {str(experiment_description.h36m_train_dataset_cfg.scale_factor).ljust(25)} |\n" \
          f"| {'Flip'.ljust(25)} | {str(experiment_description.h36m_train_dataset_cfg.flip).ljust(25)} |\n" \
          f"| {'Rotate'.ljust(25)} | {str(experiment_description.h36m_train_dataset_cfg.rotation_factor).ljust(25)} |\n"
    md += f"## General data preparation\n" \
          "```python\n" \
          "trans = transforms.Compose([\n" \
          "    transforms.ToTensor(),\n" \
          "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n" \
          "])\n```\n" \
          f"suggested lr: {experiment_description.suggested_lr:.2e}\n"

    for round_num, round in enumerate(experiment_description.experiment_rounds):
        md += f"## Round {round_num}\n" \
              f"- Epochs: **{round.num_epochs}**\n" \
              f"- LR Scheduler: **{round.scheduler.__class__.__name__}**\n" \
              f"### Optimizer\n" \
              f"| {'Property'.ljust(25)} | {'Value'.ljust(25)} |\n" \
              f"| {'-' * 25} | {'-' * 25} |\n" \
              f"| {'name'.ljust(25)} | {str(round.optimizer.__class__.__name__).ljust(25)} |\n"
        for prop, value in round.optimizer_parameters.items():
            md += f"| {prop.ljust(25)} | {str(value).ljust(25)} |\n"
        md += f"### LRs\n" \
              f"| {'Network Part'.ljust(25)} | {'LRs'.ljust(20)} | Frozen? |\n" \
              f"| {'-' * 25} | {'-' * 20} | {'-' * 7} |\n"
        for layer_idx, (layer, layer_name) in enumerate(zip(experiment_description.net_layers, experiment_description.net_layer_names)):
            lr1 = round.max_lrs[layer_idx * 2]
            lr2 = round.max_lrs[layer_idx * 2 + 1]
            md += f"| {layer_name.ljust(25)} | {f'{lr1:.2e}, {lr2:.2e}'.ljust(20)} | {str(layer in round.frozen_layers).ljust(7)} |\n"
        md += f"### Results\n" \
              f"{results_to_md_table(round.validation_results)}"
    return md


def save_log(log: str, output_path: str):
    with open(output_path, "w") as f:
        f.write(log)