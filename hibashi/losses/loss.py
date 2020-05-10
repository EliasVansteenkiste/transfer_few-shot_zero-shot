import inspect
from abc import ABC, abstractmethod

from hibashi.framework.utils import get_parameter_of_cls


class Loss(ABC):

    @staticmethod
    def all_losses():
        losses = {cls.__name__.lower(): cls for cls in Loss.__subclasses__()}
        return losses

    @staticmethod
    def get_cls(name):
        losses = Loss.all_losses()
        loss = losses.get(name.lower())
        if loss is None:
            raise ValueError('Unknown loss: "%s". Valid options are: %s' % (name.lower(), list(losses.keys())))

        return loss

    @property
    def name(self):
        return type(self).__name__

    @classmethod
    def get_loss_params(cls, name):
        loss_cls = cls.get_cls(name)
        ancestors = inspect.getmro(loss_cls)
        params = {}
        for anc in ancestors:
            params.update(get_parameter_of_cls(anc))

        return params
