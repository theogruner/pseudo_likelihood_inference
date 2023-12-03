"""Casting utilities to transform the data into desired output."""
from collections import OrderedDict, MutableMapping
from typing import Dict, Union

import numpy as np
from omegaconf import DictConfig


def dict_to_array(data_dict: OrderedDict):
    """Cast dict containing the data into a numpy array"""
    keys, out_array = list(data_dict.keys()), np.stack([data_dict.values()], axis=-1)
    return out_array, keys


def flatten_dict(dictionary: Union[DictConfig, Dict], parent_key='', sep="_") -> Dict:
    """Taken from
        `https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys`
    """
    items = []
    for key, val in dictionary.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(val, MutableMapping):
            items.extend(flatten_dict(val, new_key, sep).items())
        else:
            items.append((new_key, val))
    return dict(items)


def swapaxes(data: dict[np.array], axis1: int, axis2: int):
    """Change the axis of data contained in a dict."""
    return {key: np.swapaxes(val, axis1, axis2) for key, val in data.items()}
