"""
Numerically robust optimizer steps shared by the PedRecNet and EHPI3D training.

* mixed precision: bf16 autocast (default on GPUs with bf16 support, e.g. RTX 30xx-50xx) or fp16 with a GradScaler;
  the loss heads always run in fp32
* gradient clipping (global L2 norm) after unscaling
* gradient accumulation (effective batch size = batch size x accumulate, for GPUs with 12 GB)
* exponential moving average (EMA) of the weights incl. BatchNorm buffers
* non-finite guard: steps with a NaN / inf loss or gradient norm are skipped instead of corrupting the weights;
  training stops after too many consecutive bad steps
"""
import contextlib
import logging
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

logger = logging.getLogger(__name__)


@dataclass
class TrainingOptions:
    amp: str = "auto"  # "auto" (bf16 if supported, else fp16 on CUDA, off on CPU), "bf16", "fp16" or "off"
    grad_clip: float = 10.0  # max global gradient norm, 0 disables clipping
    accumulate: int = 1  # optimizer step every n batches
    ema_decay: float = 0.9998  # 0 disables the EMA
    max_bad_steps: int = 50  # abort after this many consecutive non-finite steps

    def describe(self) -> str:
        return (f"amp={self.amp}, grad_clip={self.grad_clip}, accumulate={self.accumulate}, "
                f"ema_decay={self.ema_decay}")


def resolve_amp_dtype(amp: str, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda" or amp == "off":
        return None
    if amp == "fp16":
        return torch.float16
    if amp == "bf16":
        return torch.bfloat16
    if amp == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    raise ValueError(f"Unknown amp mode '{amp}'")


@dataclass
class StepStats:
    optimizer_steps: int = 0
    skipped_steps: int = 0
    clipped_steps: int = 0
    grad_norms: list = field(default_factory=list)

    def summary(self) -> str:
        mean_norm = sum(self.grad_norms) / len(self.grad_norms) if self.grad_norms else float("nan")
        return (f"{self.optimizer_steps} optimizer steps, {self.skipped_steps} skipped (non-finite), "
                f"{self.clipped_steps} clipped, mean grad norm {mean_norm:.3f}")


class TrainStepper:
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, scheduler, device: torch.device,
                 options: TrainingOptions, ema: Optional[AveragedModel] = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.options = options
        self.amp_dtype = resolve_amp_dtype(options.amp, device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
        self.ema = ema
        self.stats = StepStats()
        self._consecutive_bad = 0
        self._micro_step = 0

    @staticmethod
    def create_ema(model: torch.nn.Module, options: TrainingOptions) -> Optional[AveragedModel]:
        if options.ema_decay <= 0:
            return None
        return AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(options.ema_decay), use_buffers=True)

    def autocast(self):
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype)

    def _params(self) -> Iterable[torch.nn.Parameter]:
        for group in self.optimizer.param_groups:
            yield from group["params"]

    def _bad_step(self, reason: str):
        self.optimizer.zero_grad(set_to_none=True)
        self.stats.skipped_steps += 1
        self._consecutive_bad += 1
        self._micro_step = 0
        logger.warning(f"Skipping optimizer step: {reason} ({self._consecutive_bad} in a row)")
        if self._consecutive_bad > self.options.max_bad_steps:
            raise RuntimeError(f"{self._consecutive_bad} consecutive non-finite training steps, aborting. "
                               f"Try a lower learning rate or --amp off.")

    def step(self, loss: torch.Tensor, last_batch: bool = False) -> bool:
        """
        Backward pass for one batch and, every ``accumulate`` batches, the optimizer / scheduler / EMA update.
        :return: True if an optimizer step was done
        """
        if not torch.isfinite(loss.detach()):
            self._bad_step(f"loss is {loss.item()}")
            return False
        self.scaler.scale(loss / self.options.accumulate).backward()
        self._micro_step += 1
        if self._micro_step < self.options.accumulate and not last_batch:
            return False
        self._micro_step = 0

        self.scaler.unscale_(self.optimizer)
        max_norm = self.options.grad_clip if self.options.grad_clip > 0 else math.inf
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in self._params() if p.grad is not None], max_norm)
        grad_norm = float(grad_norm)
        if not math.isfinite(grad_norm):
            if self.amp_dtype == torch.float16:
                # the GradScaler skips the step and lowers the scale (normal during the first iterations)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.stats.skipped_steps += 1
                return False
            self._bad_step(f"gradient norm is {grad_norm}")
            return False
        self._consecutive_bad = 0
        self.stats.grad_norms.append(grad_norm)
        if grad_norm > max_norm:
            self.stats.clipped_steps += 1

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update_parameters(self.model)
        self.stats.optimizer_steps += 1
        return True

    def state_dict(self) -> dict:
        return {"scaler": self.scaler.state_dict()}

    def load_state_dict(self, state: dict):
        if "scaler" in state:
            self.scaler.load_state_dict(state["scaler"])


def steps_per_epoch(num_batches: int, accumulate: int) -> int:
    """Number of optimizer (and scheduler) steps per epoch with gradient accumulation."""
    return max(1, math.ceil(num_batches / max(1, accumulate)))
