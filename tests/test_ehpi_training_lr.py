"""An inconclusive LR search must restore weights and allow the EHPI workflow to continue."""
import copy
import logging

import pytest
import torch

from pedrec.training import train_ehpi3d
from pedrec.training.experiments.ehpi3d_variants import get_variant
from pedrec.training.experiments.experiment_path_helper import get_experiment_paths
from pedrec.training.experiments.train_stepper import TrainingOptions


def _finder_class(suggested_lr):
    class FakeFinder:
        def __init__(self, model, optimizer, criterion, device):
            self.model = model
            self.original = copy.deepcopy(model.state_dict())

        def range_test(self, loader, end_lr, num_iter):
            with torch.no_grad():
                next(self.model.parameters()).add_(10)

        def plot(self, ax):
            ax.plot([1e-7, 0.9], [1, 0.5])
            return ax if suggested_lr is None else (ax, suggested_lr)

        def reset(self):
            self.model.load_state_dict(self.original)
    return FakeFinder


@pytest.mark.parametrize("suggestion", [None, 0.002])
def test_lr_search_restores_model_and_preserves_valid_suggestions(tmp_path, monkeypatch, caplog, suggestion):
    monkeypatch.setattr(train_ehpi3d, "LRFinder", _finder_class(suggestion))
    model = torch.nn.Linear(2, 1)
    before = copy.deepcopy(model.state_dict())
    plot = tmp_path / "lr.png"
    with caplog.at_level(logging.WARNING):
        lr = train_ehpi3d.find_learning_rate(model, model.parameters(), torch.nn.MSELoss(), None,
                                            torch.device("cpu"), str(plot))
    assert lr == (train_ehpi3d.DEFAULT_LR if suggestion is None else suggestion)
    assert plot.is_file()
    assert all(torch.equal(model.state_dict()[key], value) for key, value in before.items())
    assert ("using fallback max_lr" in caplog.text) == (suggestion is None)


def test_default_ehpi_training_continues_after_inconclusive_search(tmp_path, data_dir, monkeypatch):
    monkeypatch.setattr(train_ehpi3d, "LRFinder", _finder_class(None))
    paths = get_experiment_paths(str(tmp_path))
    paths.sim_c01_dir = data_dir
    paths.sim_c01_filename = "legacy_pedrec_gt_df.pkl"
    paths.sim_c01_lifted_results_filename = "legacy_pedrec_result_df.pkl"
    paths.ehpi3d_output_dir = str(tmp_path)
    checkpoint = train_ehpi3d.train_variant(get_variant("gt_pred_64frames"), paths, torch.device("cpu"),
                                            epochs=1, batch_size=8, num_workers=0,
                                            options=TrainingOptions(ema_decay=0.99))
    state = torch.load(checkpoint, weights_only=True)
    assert all(torch.isfinite(value).all() for value in state.values() if value.is_floating_point())
