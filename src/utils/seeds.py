"""Global random seed utilities."""
import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Sets random seeds across standard library, numpy, and environment variables."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
