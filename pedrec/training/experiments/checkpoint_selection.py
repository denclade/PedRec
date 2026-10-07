"""
Selection of the best epoch from the validation results of a multi task training.

The tasks have different units (PCK in %, MPJPE in mm, angles in degrees), so every metric is compared relative to
its value in the first validated epoch of the run. The score is the mean of these ratios, oriented so that lower is
better (0.9 = on average 10 % better than the reference epoch).
"""
from typing import Dict, Optional

from pedrec.models.validation.validation_results import ValidationResults

# (getter, higher_is_better)
_METRICS = {
    "pck@0.2": (lambda r: r.pose2d_pck.pck_2_mean if r.pose2d_pck is not None else None, True),
    "mpjpe": (lambda r: r.pose3d.mpjpe_mean if r.pose3d is not None else None, False),
    "joint_acc": (lambda r: r.pose2d_conf.conf_acc if r.pose2d_conf is not None else None, True),
    "body_phi_err": (lambda r: r.orientation.body.angle_error_phi if r.orientation is not None else None, False),
    "head_phi_err": (lambda r: r.orientation.head.angle_error_phi if r.orientation is not None else None, False),
}


def collect_metrics(validation_results: Dict[str, ValidationResults]) -> Dict[str, float]:
    metrics = {}
    for set_name, results in validation_results.items():
        for metric_name, (getter, _) in _METRICS.items():
            value = getter(results)
            if value is not None:
                metrics[f"{set_name}/{metric_name}"] = float(value)
    return metrics


def relative_score(metrics: Dict[str, float], reference: Dict[str, float]) -> Optional[float]:
    ratios = []
    for key, value in metrics.items():
        ref = reference.get(key)
        if ref is None:
            continue
        higher_is_better = _METRICS[key.split("/")[-1]][1]
        numerator, denominator = (ref, value) if higher_is_better else (value, ref)
        if denominator <= 1e-9:
            continue
        ratios.append(numerator / denominator)
    return sum(ratios) / len(ratios) if ratios else None


class BestEpochTracker:
    def __init__(self):
        self.reference: Optional[Dict[str, float]] = None
        self.best_score: Optional[float] = None
        self.best_epoch: Optional[str] = None

    def update(self, validation_results: Dict[str, ValidationResults], epoch_name: str) -> bool:
        """:return: True if this epoch is the best so far"""
        metrics = collect_metrics(validation_results)
        if not metrics:
            return False
        if self.reference is None:
            self.reference = metrics
        score = relative_score(metrics, self.reference)
        if score is None:
            return False
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            self.best_epoch = epoch_name
            return True
        return False

    def state_dict(self) -> dict:
        return {"reference": self.reference, "best_score": self.best_score, "best_epoch": self.best_epoch}

    def load_state_dict(self, state: dict):
        self.reference = state.get("reference")
        self.best_score = state.get("best_score")
        self.best_epoch = state.get("best_epoch")
