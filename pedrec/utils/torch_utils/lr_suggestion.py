"""
Learning rate suggestion from an LR range test (shared by LRFinder and MTLLRFinder).
"""
import math
from typing import List, NamedTuple, Optional

import numpy as np


class LRSuggestion(NamedTuple):
    lr: Optional[float]  # None: no reliable suggestion, see reason
    min_loss_lr: Optional[float]
    steepest_lr: Optional[float]
    reason: str = ""


def has_diverged(loss: float, best_loss: float, diverge_th: float) -> bool:
    """Non finite loss or loss above best + (diverge_th - 1) * |best| (= diverge_th * best for positive losses)."""
    return not math.isfinite(loss) or loss > best_loss + (diverge_th - 1) * abs(best_loss)


def suggest_lr(lrs: List[float], losses: List[float], skip_start: int = 10, skip_end: int = 5,
               min_rise: float = 0.1) -> LRSuggestion:
    """
    A tenth of the learning rate with the minimal (smoothed) loss (fastai "lr_min"), the steepest descent is only
    reported. The minimum is only meaningful if the loss rises again behind it: if it does not rise by at least
    ``min_rise`` of the loss drop until the end of the test, the tested range was too small and the "minimum" (as well
    as the steepest descent) would only be the end of the range -> no suggestion.
    """
    lrs = np.asarray(lrs, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64)
    end = len(losses) - skip_end if skip_end > 0 else len(losses)
    if end - skip_start < 3:
        return LRSuggestion(None, None, None, f"only {len(losses)} iterations, too few for a suggestion")
    min_idx = skip_start + int(np.argmin(losses[skip_start:end]))
    steepest_idx = skip_start + int(np.argmin(np.gradient(losses[skip_start:min_idx + 1]))) \
        if min_idx - skip_start >= 2 else min_idx
    min_loss_lr, steepest_lr = float(lrs[min_idx]), float(lrs[steepest_idx])
    drop = losses[skip_start] - losses[min_idx]
    rise = np.max(losses[min_idx:]) - losses[min_idx]
    if drop <= 0:
        return LRSuggestion(None, min_loss_lr, steepest_lr, "the loss did not decrease")
    if rise < min_rise * drop:
        return LRSuggestion(None, min_loss_lr, steepest_lr,
                            f"the loss still decreases at the end of the tested range (lr {lrs[-1]:.2e}), "
                            f"increase end_lr")
    return LRSuggestion(min_loss_lr / 10, min_loss_lr, steepest_lr)
