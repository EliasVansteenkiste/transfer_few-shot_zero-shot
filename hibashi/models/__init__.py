import importlib
import os
import pkgutil
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any

import numpy as np
import re
import torch
from torch.optim import Optimizer

from hibashi.framework.utils import get_subclass, get_parameter_of_cls


class Model(ABC):
    NETWORK_PREFIX = 'net_'
    LOSS_PREFIX = 'loss_'
    METRIC_PREFIX = 'metric_'
    NON_SCALAR_METRIC_PREFIX = 'non_scalar_metric_'
    OPTIMIZER_PREFIX = 'optim_'
    CRITERION_PREFIX = 'criterion_'
    INPUT_PREFIX = 'in_'
    OUTPUT_PREFIX = 'out_'

    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.

    The base class follows the principle of convention over configuration. Specific objects like networks,
    losses and optimizer should have variable names with special prefixes. The prefixes can be overwritten in subclasses
    but unless their is a special need for it, it's highly recommended to stick to the defaults. 
    """

    def __init__(self, gpu_ids, is_train=True):
        """Initialize the BaseModel class.
        Parameters:
        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        """
        self.gpu_ids = gpu_ids
        self.is_train = is_train
        # get device name: CPU or GPU
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids != -1 else torch.device('cpu')
        self.training = True
        self.date_created = datetime.now()

    @property
    @abstractmethod
    def name_main_metric(self):
        pass

    @abstractmethod
    def main_metric(self, metrics):
        pass

    @property
    def network_names(self) -> List[str]:
        networks_names = list(self.networks.keys())
        return networks_names

    @property
    def networks(self) -> Dict[str, torch.nn.Module]:
        networks = {}

        for k, v in self.__dict__.items():
            if k.startswith(type(self).NETWORK_PREFIX):
                networks[k] = v

        return networks

    @property
    def optimizers(self) -> Dict[str, Optimizer]:
        optimizers = {}

        for k, v in self.__dict__.items():
            if k.startswith(type(self).OPTIMIZER_PREFIX):
                optimizers[k] = v

        return optimizers

    @property
    def loss_names(self) -> List[str]:
        loss_names = list(self.losses.keys())
        return loss_names

    @property
    def losses(self) -> Dict[str, Any]:
        losses = {}
        for k, v in self.__dict__.items():
            if k.startswith(type(self).LOSS_PREFIX):
                losses[k] = v

        return losses

    @property
    def metric_names(self) -> List[str]:
        metric_names = list(self.metrics.keys())
        return metric_names

    @property
    def metrics(self) -> Dict[str, Any]:
        metrics = {}
        for k, v in self.__dict__.items():
            if k.startswith(type(self).LOSS_PREFIX) or k.startswith(type(self).METRIC_PREFIX):
                metrics[k] = v

        return metrics

    @property
    def non_scalar_metrics_names(self) -> List[str]:
        non_scalar_metric_names = list(self.non_scalar_metrics.keys())
        return non_scalar_metric_names

    @property
    def non_scalar_metrics(self) -> Dict[str, Any]:
        non_scalar_metrics = {}
        for k, v in self.__dict__.items():
            if k.startswith(type(self).NON_SCALAR_METRIC_PREFIX):
                non_scalar_metrics[k] = v

        return non_scalar_metrics

    @property
    def state(self) -> Dict[str, Any]:
        state = {}
        for k, v in self.__dict__.items():
            if k.startswith(type(self).LOSS_PREFIX) or \
               k.startswith(type(self).METRIC_PREFIX) or \
               k.startswith(type(self).INPUT_PREFIX) or \
               k.startswith(type(self).NON_SCALAR_METRIC_PREFIX) or \
               k.startswith(type(self).OUTPUT_PREFIX):
                # if v is a tensor leave behind the grads, we only need the value
                if type(v) == torch.Tensor:
                    state[k] = v.data
                else:
                    state[k] = v

        return state

    @property
    def learning_rates(self) -> Dict[str, Any]:
        lrs = {}
        for k, v in self.__dict__.items():
            if k.startswith(type(self).OPTIMIZER_PREFIX):
                lrs[k + '_lr'] = v.param_groups[0]['lr']

        return lrs

    @property
    def visuals(self) -> Dict[str, np.array]:
        return {}

    @property
    def figures(self):
        return {}

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def get_validation_figures(self, state: dict) -> dict:
        """Gets validation figures"""
        return {}

    def eval(self):
        """Make models eval mode during test time"""

        if not self.training:
            return

        for net in self.networks.values():
            net.eval()
        self.training = False

    def train(self):
        """Make models train mode"""

        if self.training:
            return

        for net in self.networks.values():
            net.train()
        self.training = True

    def test(self):
        """Forward function used in test time.
        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop and
        calculates all the metrics.
        """
        with torch.no_grad():
            self.forward()
            self.calculate_metrics()

    def infer(self, inp):
        """Used during inference time"""
        self.eval()
        with torch.no_grad():
            return self._infer(inp)

    def _infer(self, inp):
        raise NotImplementedError

    def calculate_metrics(self):
        """Calculate metrics which are used during evaluation. This should also include all the losses"""
        pass

    def load_networks(self, path, epoch='best'):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name, net in self.networks.items():
            if epoch == 'best':
                pattern = 'best_%s_.*\.pth' % name
                for i in os.listdir(path):
                    if os.path.isfile(os.path.join(path, i)) and re.match(pattern, i):
                        load_filename = i
                        break
                else:
                    raise ValueError('No best weights found for network %s' % name)
                epoch = int(load_filename.split('_')[3])
            else:
                load_filename = '{prefix}_{name}_{epoch}.pth'.format(prefix='last', name=name, epoch=epoch)

            load_path = os.path.join(path, load_filename)

            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)

            state_dict = torch.load(load_path, map_location=self.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net.load_state_dict(state_dict)
        return

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture
        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name in self.network_names:
            if isinstance(name, str):
                net = getattr(self, name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    @classmethod
    def get_default_params(cls, name):
        subclass = get_subclass(name, cls)
        params = get_parameter_of_cls(subclass, ancestors=False)

        return params


for (module_loader, name, ispkg) in pkgutil.iter_modules([os.path.split(__file__)[0]]):
    print('hibashi.models.' + name + '.losses', __package__)
    print('hibashi.models.' + name + '.' + name, __package__)
    importlib.import_module('hibashi.models.' + name + '.losses', __package__)
    importlib.import_module('hibashi.models.' + name + '.' + name, __package__)