from contextlib import contextmanager
import torch
import numpy as np


@contextmanager
def temp_seed(seed: int, library: str):
    """
    Sets a temporal local seed to do a specific task.
    Example: with temp_seed(int): do something.
    :param seed: temporal local seed to set
    :param library: library containing the implementation of the random generator to be set
    Either numpy or torch
    :return:
    """

    if library == 'numpy':
        original_state = np.random.get_state()
        np.random.seed(seed)
        try:
            yield
        finally:
            np.random.set_state(original_state)
    elif library == 'torch':
        original_state = torch.random.get_rng_state()
        torch.random.manual_seed(seed)
        try:
            yield
        finally:
            torch.random.set_rng_state(original_state)
    else:
        raise NotImplementedError






