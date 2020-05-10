from torch.optim.optimizer import Optimizer, required
from torch.optim import Optimizer

from hibashi.framework.utils import get_subclass, get_parameter_of_cls


def create_optimizer(name, params, model):
    opt_cls = get_subclass(name, Optimizer)

    return opt_cls(model.parameters(), **params)


def get_optimizer_params(name):
    opt_cls = get_subclass(name, Optimizer)
    params = get_parameter_of_cls(opt_cls)

    return params


class MultipleOptimizer(object):
    """
    Performs zero_grad and step for multiple optimizers.
    """

    def __init__(self, *optimizer):
        """
        :param optimizer: (pytorch object) with an optimizer
        """
        self.optimizers = optimizer

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()
