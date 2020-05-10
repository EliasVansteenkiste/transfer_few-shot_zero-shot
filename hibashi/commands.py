import imgaug
import importlib
import numpy as np
import random
from sacred.commandline_options import CommandLineOption
import torch
from torch.backends import cudnn


from hibashi.framework.utils import get_parameter_of_cls, all_subclasses


def print_options(options, with_defaults=True):
    for option in options:
        msg = '\t' + option.__name__

        if with_defaults:
            defaults = get_parameter_of_cls(option)
            msg += ''.join(['\n\t* %s=%s' % (k, v) for k, v in defaults.items()])

        msg += '\n'
        print(msg)


def models():
    """
    print overview of all models
    """
    print('\n', 'Models:')
    from hibashi.models import Model as M
    models = all_subclasses(M)
    print_options(models)


def networks():
    """
    print overview of all networks
    """
    print('\n', 'Networks:')
    from hibashi.networks.network import Network as N
    networks = all_subclasses(N)
    print_options(networks)


def datasets():
    """
    print overview of all datasets
    """
    print('\n', 'Datasets:')
    from hibashi.data.datasets.datasets import Dataset as D
    datasets = all_subclasses(D)
    print_options(datasets)


def losses():
    """
    print overview of all losses
    """
    print('\n', 'Losses:')
    from hibashi.losses.loss import Loss as L
    losses = all_subclasses(L)
    print_options(losses)


def optimizers():
    """
    print overview of all optimizers
    """
    print('\n', 'Optimizers:')
    from torch.optim import Optimizer as O
    optimizers = all_subclasses(O)
    print_options(optimizers)


class CudnnBenchmark(CommandLineOption):
    """
    Set's torch.backends.cudnn.benchmark = True this is can improve memory usage and speed.
    """
    short_flag = 'cb'


def cudnn_hook(options):
    if options['--cudnn_benchmark'] is True:
        cudnn.benchmark = True
    else:
        cudnn.benchmark = False


class Model(CommandLineOption):
    """
    Name of the model to use
    """
    arg = 'Hibashi'


def ingredient_hook(ex):
    print('building ingredient_hook')

    def _func(options):
        print('ingredient_hook triggered')
        model = options['--model']
        # first find the seed
        for update in options['UPDATE']:
            if 'seed=' in update:
                seed = int(update.replace('seed=', ''))
                setting_seeds(seed)

        ex.path = model
        opt = importlib.import_module(f'hibashi.models.{model}.options')
        ex.ingredients.append(opt.train)
        ex.ingredients.append(opt.train_data)
        ex.ingredients.append(opt.val_data)
    return _func


def setting_seeds(seed):
    print('Setting seed to: %s' % seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    imgaug.seed(seed)


def add_commands(ex):
    ex.command(models, unobserved=True)
    ex.command(networks, unobserved=True)
    ex.command(datasets, unobserved=True)
    ex.command(losses, unobserved=True)
    ex.command(optimizers, unobserved=True)

    ex.option_hook(cudnn_hook)
    ex.option_hook(ingredient_hook(ex))
