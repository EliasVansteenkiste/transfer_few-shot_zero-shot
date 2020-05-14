import torch
from torch.nn import functional as tnnf

from hibashi.losses.loss import Loss


class CE(Loss):
    def __init__(self):
        super(CE, self).__init__()

    def __call__(self, pred, target, weight=None):
        return tnnf.cross_entropy(pred, target, weight=weight, reduction='mean')


class Accuracy(Loss):
    def __init__(self):
        super(Accuracy, self).__init__()

    def __call__(self, pred, target, weight=None):
        batch_size = target.size(0)
        values, indices = pred.max(1)
        return torch.sum(indices == target).float()/batch_size
