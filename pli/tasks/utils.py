from typing import Dict, List, Tuple

import numpy as np


def support_dict_to_array(
    param_support: Dict[str, Tuple[float, float]], param_names: List[str]
) -> np.ndarray:
    """Utility to get a tensorized representation of the support of the parameters
    :param param_support: Dict with lower and upper support entries
    :param param_names: A list of named ordered parameters
    :returns: Support ranges with Dim[2, n_params].
    The first dimension denotes lower and upper value, of the individual parameters.
    """
    lower_support = np.array([param_support[key][0] for key in param_names], dtype=np.float_)
    upper_support = np.array([param_support[key][1] for key in param_names], dtype=np.float_)
    return np.stack([lower_support, upper_support])
