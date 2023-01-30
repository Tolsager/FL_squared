import os

import pytest

from src.utils import load_config


@pytest.mark.skipif(
    not os.path.exists("configs/train_gpu.yaml"), reason="config file does not exist"
)
def test_load_config():
    config1 = load_config("train_gpu")
    config2 = load_config("train_gpu.yaml")

    assert isinstance(config1, dict)
    assert isinstance(config2, dict)

    assert config1 == config2
