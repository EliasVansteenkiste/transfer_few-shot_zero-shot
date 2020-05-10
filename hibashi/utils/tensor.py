import torch


def warn_if_nan(x, name):
    if torch.isnan(x).any():
        print(f'{name} is nan')