"""
Collects PedRecNet validation results in doc/experiments/results.csv (one value per row) and prints comparison tables.

    # add the last epoch of a training protocol (or --epochs all / --epochs R2E28)
    python pedrec/tools/experiments/collect_results.py add data/models/pedrec/single_results/experiment_pedrec_v2_p2d_c_protocol.md \\
        --run v2-2026-10-28 --model v2 --stage p2d_c --note "..."
    # compare runs
    python pedrec/tools/experiments/collect_results.py compare --metrics "COCO PCK@0.05" "H36M MPJPE"

Rows of the same run / stage / checkpoint / dataset / metric are replaced. Units: PCK in %, MPJPE in mm, PCJP / PCD as
fraction, orientation phi in degrees (the protocols store radians), spherical distances as in the protocol.
"""
import sys

sys.path.append('.')  # allow running as a script from the repository root

import argparse
import csv
import math
import os
import re
from typing import Dict, List, Optional

DEFAULT_CSV = os.path.join("doc", "experiments", "results.csv")
COLUMNS = ["run", "model", "stage", "checkpoint", "dataset", "metric", "value", "note"]

# protocol column suffix -> (metric name, conversion)
_METRICS = {
    "PCK@0.05": ("PCK@0.05", None),
    "PCK@0.2": ("PCK@0.2", None),
    "MPJPE": ("MPJPE", None),
    "MRCJP": ("PCJP", None),
    "MRCD": ("PCD", None),
    "JointAcc": ("JointAcc", None),
    "Body Sph.Dist.": ("Body sph", None),
    "Head Sph.Dist.": ("Head sph", None),
    "Body.Phi.Ang.Dist.": ("Body phi deg", math.degrees),
    "Head.Phi.Ang.Dist.": ("Head phi deg", math.degrees),
}


def parse_protocol(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    {checkpoint: {dataset: {metric: value}}} of all epochs of a training protocol; checkpoints are named like the
    training log (R2E43 = round 2 of 2, epoch 43, 1-based). Columns without a known metric (losses, times) are skipped.
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    round_idx: Optional[int] = None
    header: Optional[List[str]] = None
    with open(path) as f:
        for line in f:
            match = re.match(r"## Round (\d+)", line)
            if match:
                round_idx, header = int(match.group(1)), None
                continue
            if not line.startswith("|") or round_idx is None:
                continue
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if cells[0] == "Epoch":
                header = cells
                continue
            if header is None or not cells[0].isdigit() or len(cells) != len(header):
                continue
            checkpoint = f"R{round_idx + 1}E{int(cells[0]) + 1}"
            for column, cell in zip(header[1:], cells[1:]):
                dataset, _, suffix = column.partition(" ")
                if suffix not in _METRICS or cell in ("", "-"):
                    continue
                metric, convert = _METRICS[suffix]
                value = float(cell)
                results.setdefault(checkpoint, {}).setdefault(dataset, {})[metric] = \
                    convert(value) if convert else value
    return results


def _checkpoint_key(checkpoint: str):
    match = re.match(r"R(\d+)E(\d+)", checkpoint)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def read_rows(csv_path: str) -> List[dict]:
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(csv_path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def merge_rows(existing: List[dict], new: List[dict]) -> List[dict]:
    """New rows replace existing ones with the same run / stage / checkpoint / dataset / metric."""
    def key(row):
        return row["run"], row["stage"], row["checkpoint"], row["dataset"], row["metric"]
    new_keys = {key(row) for row in new}
    return [row for row in existing if key(row) not in new_keys] + new


def protocol_rows(path: str, run: str, model: str, stage: str, epochs: str, note: str = "") -> List[dict]:
    results = parse_protocol(path)
    if not results:
        raise ValueError(f"No epoch results found in {path}")
    checkpoints = sorted(results, key=_checkpoint_key)
    if epochs == "last":
        selected = [checkpoints[-1]]
    elif epochs == "all":
        selected = checkpoints
    else:
        selected = [c for c in epochs.split(",") if c in results]
        missing = set(epochs.split(",")) - set(selected)
        if missing:
            raise ValueError(f"Checkpoints {sorted(missing)} not in {path} (available: {checkpoints[0]} ... "
                             f"{checkpoints[-1]})")
    rows = []
    for checkpoint in selected:
        label = f"{checkpoint} (last)" if checkpoint == checkpoints[-1] else checkpoint
        for dataset, metrics in results[checkpoint].items():
            for metric, value in metrics.items():
                rows.append({"run": run, "model": model, "stage": stage, "checkpoint": label, "dataset": dataset,
                             "metric": metric, "value": f"{value:.4g}", "note": note})
    return rows


def comparison_table(rows: List[dict], metrics: List[str], runs: Optional[List[str]] = None) -> str:
    """Markdown table: one column per run / stage / checkpoint, one row per 'DATASET METRIC'."""
    columns = []
    for row in rows:
        column = (row["run"], row["stage"], row["checkpoint"])
        if (runs is None or row["run"] in runs) and column not in columns:
            columns.append(column)
    values = {(row["run"], row["stage"], row["checkpoint"], f"{row['dataset']} {row['metric']}"): row["value"]
              for row in rows}
    columns = [(r, s, c) for r, s, c in columns if any((r, s, c, metric) in values for metric in metrics)]
    lines = ["| Metric | " + " | ".join(f"{r} {s} {c}" for r, s, c in columns) + " |",
             "| --- |" + " --- |" * len(columns)]
    for metric in metrics:
        cells = [values.get((r, s, c, metric), "") for r, s, c in columns]
        if any(cells):
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="Add the results of a training protocol (markdown).")
    add.add_argument("protocol")
    add.add_argument("--run", required=True, help="Run id, e.g. v2-2026-10-28 (date of the training start).")
    add.add_argument("--model", required=True, help="v1 / v2")
    add.add_argument("--stage", required=True, help="Training stage, e.g. p2d_c / p2d3d_c_o")
    add.add_argument("--epochs", default="last", help="last (default), all or checkpoints like R2E28,R2E30")
    add.add_argument("--note", default="", help="Comparability note, e.g. 'SIM val = training data'")
    compare = commands.add_parser("compare", help="Print a markdown comparison table.")
    compare.add_argument("--metrics", nargs="+", required=True, help="'DATASET METRIC', e.g. 'H36M MPJPE'")
    compare.add_argument("--runs", nargs="*", default=None, help="Only these runs (default: all)")
    args = parser.parse_args(argv)

    if args.command == "add":
        new = protocol_rows(args.protocol, args.run, args.model, args.stage, args.epochs, args.note)
        write_rows(args.csv, merge_rows(read_rows(args.csv), new))
        print(f"{len(new)} values of {args.protocol} written to {args.csv}")
    else:
        print(comparison_table(read_rows(args.csv), args.metrics, args.runs))


if __name__ == "__main__":
    main()
