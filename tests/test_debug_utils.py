import copy

import torch

from src import debug_utils
from src.models.resnet import ResNet18


def test_frac_grad():
    model = ResNet18()
    frac = debug_utils.frac_grad(model)
    assert isinstance(frac, float)


def test_are_models_identical():
    with torch.no_grad():
        m1 = ResNet18()
        m2 = copy.deepcopy(m1)
        assert debug_utils.are_models_identical(m1, m2)
        m2.bn1.weight[0] = 2
        assert not debug_utils.are_models_identical(m1, m2)
