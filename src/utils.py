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
