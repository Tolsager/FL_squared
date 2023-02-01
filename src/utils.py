import os
import random

import numpy as np
import torch
import yaml
from yaml.loader import SafeLoader


def load_yaml(file_path: str) -> dict:
    """safely loads a yaml file and returns
    it as a dict

    Args:
        file_path (str): path to the yaml file with extension

    Returns:
        dict: key-value pairs of the yaml
    """
    with open(file_path, "r") as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def load_config(name: str) -> dict:
    """loads the config file from the name of the file
    alone. The .yaml extension is optional.

    Args:
        name (str): name of the config file

    Returns:
        dict: the nested config dictionary
    """
    config_path = name + ".yaml" if not name.endswith(".yaml") else name
    config = load_yaml("configs/" + config_path)

    return config


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
