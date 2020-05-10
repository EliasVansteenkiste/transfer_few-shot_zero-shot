import inspect
import sys
from collections import defaultdict
from typing import Dict, Union

import decorator

from hibashi.framework.utils import get_parameter_of_cls, get_args, get_subclass

# stores all the factories so they can later be injected with the correct parameters
factories = defaultdict(dict)


def factory(name, base_class, default_class, **kwargs):
    """
    Decorator factory for a factory decorator which allows to inject configurable objects into a models ini function.
    The injected objects will be accessible through the kwargs argument through the defined name. In cases the factory
    class requires arguments for it's instantiation the decorator will return a factory method instead of an object.
    The factory method will have all the parameters inluded inside and only requires the argument to build the final
    object. This is for example needed for optimizers since they require model parameters.

    Example:
        @factory('network', Network, ResFCN256)
        @factory('loss', Loss, 'WeightMaskMSE')
        @factory('optim', Optimizer, 'adam', lr=0.0001, weight_decay=0.0)
        def __init__(self, gpu_ids, is_train, **kwargs):
            network = kwargs['network']
            loss = kwargs['loss']
            optim = kwargs['optim']

            # implement some magic

    :param name: name of the factory. will determent the key inside the kwargs dict and also the name in the sacred conf
    :param base_class: Base class of the factory object. For example Network for all the network definitions.
    :param default_class: The default class to use if not overwritten by the config. Can be string or type
    :param kwargs: Additional default parameters for the factory to use if not overwritten by the config
    :return:
    """
    model = inspect.stack()[1].function

    base_class = base_class
    name = name

    cls = get_subclass(default_class, base_class)
    params = get_parameter_of_cls(cls)
    params.update(**kwargs)
    has_args = len(get_args(cls)) > 0

    factories[model.lower()][name] = {
        'params': params,
        'cls': cls,
        'has_args': has_args,
        'base_class': base_class
    }

    def _func(caller, *args, **kwargs):
        """
        The actual decorator function which gets returned by the decorator factory.
        :param caller: the function which gets wrapped by the decorator
        :param args:
        :param kwargs:
        :return:
        """
        _factory = factories[model.lower()][name]
        try:
            if _factory['has_args']:
                def _cls(*args):
                    return _factory['cls'](*args, **_factory['params'])

                obj = _cls
            else:
                obj = _factory['cls'](**_factory['params'])
        except TypeError as e:
            msg = str(e)
            msg += '\nValid options for class %s are:' % _factory['cls'].__name__
            msg += ''.join(['\n\t* %s (default=%s)' % (k, v) for k, v in get_parameter_of_cls(_factory['cls']).items()])
            raise type(e)(msg).with_traceback(sys.exc_info()[2])

        kwargs[name] = obj
        return caller(*args, **kwargs)

    return decorator.decorator(_func)


def update_factory_class(model_name: str, factory_name: str, new_class: Union[str, type]) -> None:
    """
    Updates one factory of a given model depending on a chosen class which can be a string which matches the
    class name or a type itself.
    :param model_name:
    :param factory_name:
    :param new_class: str or class
    :return:
    """
    if factory_name not in factories[model_name].keys():
        return

    _factory = factories[model_name][factory_name]
    _factory['cls'] = get_subclass(new_class, _factory['base_class'])

    params = get_parameter_of_cls(_factory['cls'])
    _factory['params'] = params
    _factory['has_args'] = len(get_args(_factory['cls'])) > 0


def update_factories(model_name: str, model_config: Dict[str, str]) -> None:
    """
    Updates all factories of a given model based on the values set in the given model config
    :param model_name:
    :param model_config:
    :return:
    """

    for k, v in model_config.items():
        update_factory_class(model_name, k, v)


def get_model_factory_class_names(model_name: str) -> Dict[str, str]:
    """
    Collects all the factory class names for a given model and returns it in a dictionary with the factory name as key.
    :param model_name: model name
    :return:
    """
    model_factories = factories[model_name]
    defaults = {k: v['cls'].__name__ for k, v in model_factories.items()}
    return defaults


def update_params(model_name: str, config: dict) -> None:
    """
    Updates the parameter of a factory which will later be injected based on a dictionary config.
    :param config:
    :return:
    """
    model_factories = factories[model_name]
    for k, v in config.items():
        _factory = model_factories.get(k)
        if _factory is None:
            continue

        _factory['params'] = v


def get_model_factory_params(model_name: str) -> dict:
    """
    Returns a dictionary with all the factory names and default values for a given model
    :param model_name: model name as string
    :return:
    """
    model_factories = factories[model_name]
    defaults = {k: v['params'] for k, v in model_factories.items() if len(v['params']) > 0}
    return defaults
