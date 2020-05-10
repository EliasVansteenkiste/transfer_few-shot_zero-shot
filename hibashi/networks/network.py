from abc import ABC

import inspect
from torch import nn

from hibashi.framework.utils import all_subclasses, get_parameter_of_cls


class Network(nn.Module, ABC):

    def __init__(self, **kwargs):
        super(Network, self).__init__()

    @classmethod
    def all_models(cls):
        subclasses = all_subclasses(cls)
        models = {cls.__name__.lower(): cls for cls in subclasses}
        return models

    @classmethod
    def get_cls(cls, name):
        models = cls.all_models()
        model = models.get(name.lower())
        if model is None:
            raise ValueError('Unknown model: "%s". Valid options are: %s' % (name.lower(), list(models.keys())))

        return model

    @classmethod
    def get_model_params(cls, name):
        model_cls = cls.get_cls(name)
        ancestors = inspect.getmro(model_cls)
        params = {}
        for anc in ancestors:
            params.update(get_parameter_of_cls(anc))

        return params

    @property
    def name(self):
        return type(self).__name__