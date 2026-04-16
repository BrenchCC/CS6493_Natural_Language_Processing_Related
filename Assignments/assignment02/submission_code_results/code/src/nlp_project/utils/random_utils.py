import random

import torch


def set_seed(seed):
    """Set random seeds.

    Parameters:
        seed (int): Global random seed.

    Returns:
        None.
    """
    random.seed(seed)
    torch.manual_seed(seed)
