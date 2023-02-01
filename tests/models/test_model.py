import pytest
import torch

from src.models import model


def test_forward():
    cnn = model.ClientCNN(learning_rate=1)
    test_input = torch.rand(7, 3, 32, 32)
    assert cnn(test_input).shape == torch.Size([7, 10])
