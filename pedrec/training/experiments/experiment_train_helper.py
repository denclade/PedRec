import logging
import math
import time
from typing import Callable, List, Dict, Tuple, Optional

import torch.nn as nn
import torch.utils.data.distributed
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from pedrec.evaluations.validate import validate, ValidationResults
from pedrec.models.data_structures import ImageSize
from pedrec.models.experiments.epoch_validation_results import EpochValidationResults
from pedrec.models.experiments.experiment_description import ExperimentDescription
from pedrec.models.experiments.experiment_round_description import ExperimentRoundDescription
from pedrec.models.experiments.validation_set_model import ValidationSet
from pedrec.training.experiments.experiment_log_helper import log_epoch_validation_results
from pedrec.utils.log_helper import configure_logger
from pedrec.training.experiments.train_stepper import TrainStepper, TrainingOptions
from pedrec.utils.torch_utils.torch_helper import move_to_device, set_fixed_seeds, unfreeze_layers, freeze_layers
from torch.optim.swa_utils import AveragedModel

logger = logging.getLogger(__name__)


def get_subsampled_dataset(dataset: Dataset, subsample: int) -> Dataset:
    if subsample == 1:
        return dataset
    dataset_len = len(dataset)
    subsampled_dataset = Subset(dataset, list(range(0, dataset_len, subsample)))
    print(f"train full size: {dataset_len}, subsampled: {len(subsampled_dataset)}")
    return subsampled_dataset


def get_preds_single(outputs: torch.Tensor):
    """
    Helper for single models, outputs outputs[0] as every possible gt
    """
    output = outputs.cpu().detach().numpy()
    return {
        "skeleton": output,
        "skeleton_3d": output,
        "orientation": output,
    }


def get_preds_mtl(outputs: torch.Tensor):
    orientation = None
    if len(outputs) > 2:
        orientation = outputs[3].cpu().detach().numpy()
    return {
        "skeleton": outputs[0].cpu().detach().numpy(),
        "skeleton_3d": outputs[1].cpu().detach().numpy(),
        "orientation": orientation
    }



def get_outputs_loss_single(net: nn.Module, model_input: torch.Tensor, labels: torch.Tensor, loss_func: Callable):
    outputs = net(model_input)
    loss = loss_func(outputs, labels)
    return outputs, loss


def get_outputs_loss_mtl(net: nn.Module, model_input: torch.Tensor, labels: torch.Tensor):
    return net(model_input, labels)


def train(net: nn.Module, stepper: TrainStepper, train_loader: DataLoader, device: torch.device,
          get_outputs_loss_func: Callable) -> Tuple[float, float]:
    """One epoch. Returns (mean loss of the finite steps, duration in seconds)."""
    start = time.time()
    loss_total = 0.0
    finite_batches = 0
    net.train()
    num_batches = len(train_loader)
    with tqdm(total=num_batches) as pbar:
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = move_to_device(labels, device)
            with stepper.autocast():
                outputs, loss = get_outputs_loss_func(net, inputs, labels)
            loss_value = loss.item()
            stepper.step(loss, last_batch=i == num_batches - 1)
            if math.isfinite(loss_value):
                loss_total += loss_value
                finite_batches += 1
            pbar.set_postfix(loss=f"{loss_value:.4f}")
            pbar.update()
    logger.info(f"Epoch steps: {stepper.stats.summary()}")
    return loss_total / max(finite_batches, 1), time.time() - start


EpochCallback = Callable[[int, nn.Module, EpochValidationResults, TrainStepper], None]


def train_round(net: nn.Module, experiment_description: ExperimentDescription,
                experiment_round_description: ExperimentRoundDescription, train_loader: DataLoader,
                validation_sets: List[ValidationSet], get_outputs_loss_func: Callable, get_gt_pred_func: Callable,
                device: torch.device, log: bool = True, freeze_hard: bool = False,
                options: TrainingOptions = None, ema: Optional[AveragedModel] = None,
                start_epoch: int = 0, on_epoch_end: Optional[EpochCallback] = None,
                stepper_state: Optional[dict] = None) -> List[EpochValidationResults]:
    """
    Trains one round (fixed set of frozen layers, one optimizer / OneCycle schedule).

    The scheduler is stepped after every optimizer step. Validation uses the EMA weights if an EMA is given.
    ``on_epoch_end(epoch, eval_net, results, stepper)`` is called after the validation of every epoch (checkpoints).
    """
    options = options or TrainingOptions()
    unfreeze_layers(net.children())
    for layer in experiment_round_description.frozen_layers:
        freeze_layers(layer.children(), hard=freeze_hard)
    stepper = TrainStepper(net, experiment_round_description.optimizer, experiment_round_description.scheduler,
                           device, options, ema=ema)
    if stepper_state is not None:
        stepper.load_state_dict(stepper_state)
    epoch_results: List[EpochValidationResults] = []
    for epoch in range(start_epoch, experiment_round_description.num_epochs):
        train_loss, train_time = train(net, stepper, train_loader, device, get_outputs_loss_func)
        eval_net = ema.module if ema is not None else net
        validation_results = validate_val_sets(eval_net, validation_sets, get_outputs_loss_func, get_gt_pred_func,
                                               device, experiment_description.net_cfg.model.input_size,
                                               udp=experiment_description.net_cfg.arch.udp)
        results = EpochValidationResults(epoch=epoch, train_loss=train_loss, train_time=train_time,
                                         validation_results=validation_results)
        epoch_results.append(results)
        if log:
            log_epoch_validation_results(results)
        if on_epoch_end is not None:
            on_epoch_end(epoch, eval_net, results, stepper)
    return epoch_results



################## Validation Methods ####################
def validate_val_sets(net, validation_sets: List[ValidationSet], get_outputs_loss_func: Callable,
                     get_preds_func: Callable, device: torch.device, model_input_size: ImageSize,
                     udp: bool = False) -> Dict[str, ValidationResults]:
    results: Dict[str, ValidationResults] = {}
    for val_set in validation_sets:
        skeleton_3d_range = 0
        if val_set.val_set_cfg is not None:
            skeleton_3d_range = val_set.val_set_cfg.skeleton_3d_range
        val_result = validate(net, val_set.loader, get_outputs_loss_func, get_preds_func, device, model_input_size,
                              validate_2D=val_set.validate_2D,
                              validate_3D=val_set.validate_3D,
                              validate_orientation=val_set.validate_orientation,
                              validate_pose_conf=val_set.validate_pose_conf,
                              validate_env_position=val_set.validate_env_position,
                              skeleton_3d_range=skeleton_3d_range, udp=udp)
        results[val_set.name] = val_result
    return results

def init_experiment(seed: int):
    set_fixed_seeds(seed)
    configure_logger()
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True