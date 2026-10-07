import numpy as np
import torch
from torch.utils.data import TensorDataset

from pedrec.datasets.augmentations import half_body_center_scale, color_jitter, random_erasing, UPPER_IDX, LOWER_IDX
from pedrec.models.data_structures import ImageSize
from pedrec.training.experiments.experiment_dataset_helper import get_dataset_balancing_sampler


def test_half_body_crop_is_smaller_than_the_person():
    rng = np.random.default_rng(0)
    skeleton = np.zeros((26, 5), dtype=np.float32)
    skeleton[:, 0] = 400 + rng.random(26) * 100
    skeleton[UPPER_IDX, 1] = np.linspace(100, 340, len(UPPER_IDX))  # head to hands
    skeleton[LOWER_IDX, 1] = np.linspace(360, 600, len(LOWER_IDX))  # hips to feet
    skeleton[:, 3] = 1
    result = half_body_center_scale(skeleton, ImageSize(192, 256))
    assert result is not None
    _, scale = result
    assert scale[1] < 500 * 1.25
    skeleton[:, 3] = 0
    assert half_body_center_scale(skeleton, ImageSize(192, 256)) is None


def test_color_jitter_and_random_erasing():
    img = np.full((256, 192, 3), 128, dtype=np.uint8)
    assert color_jitter(img, 0) is img
    jittered = color_jitter(img, 0.5)
    assert jittered.dtype == np.uint8 and jittered.shape == img.shape
    tensor = torch.zeros(3, 256, 192)
    assert torch.count_nonzero(random_erasing(tensor, 1.0)) > 0
    assert torch.count_nonzero(random_erasing(tensor, 0.0)) == 0


def test_dataset_balancing_sampler():
    big, small = TensorDataset(torch.zeros(900)), TensorDataset(torch.ones(100))
    sampler = get_dataset_balancing_sampler([big, small], ["sim", "coco"], {"sim": 1, "coco": 1})
    torch.manual_seed(0)
    indices = torch.tensor(list(iter(sampler)))
    share_small = (indices >= 900).float().mean().item()
    assert 0.4 < share_small < 0.6  # equal probability for both datasets despite the 9:1 size ratio
