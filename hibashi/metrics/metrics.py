from abc import ABC

from ignite.exceptions import NotComputableError
from ignite.metrics import Metric


class LossFromDict(Metric):
    """
    Calculates the average loss from metric from a dictionary with a given key/name.

    Args:
        loss_name (str): key of the loss in the output dictionary
        output_transform (callable): a callable that is used to transform the
            :class:`~ignite.engine.Engine`'s `process_function`'s output into the
            form expected by the metric.
            This can be useful if, for example, you have a multi-output model and
            you want to compute the metric with respect to one of the outputs.
            The output is is expected to be a tuple (prediction, target) or
            (prediction, target, kwargs) where kwargs is a dictionary of extra
            keywords arguments.

    """

    def __init__(self, loss_name, output_transform=lambda x: x, reduce=True):
        super(LossFromDict, self).__init__(output_transform)
        self._reduce = reduce
        self._loss_name = loss_name
        self._sum = 0
        self._num_steps = 0

    def reset(self):
        self._sum = 0
        self._num_steps = 0

    def update(self, output):
        self._sum += output[self._loss_name]
        self._num_steps += 1

    def compute(self):
        if self._num_steps == 0:
            raise NotComputableError(f'{type(self)} Loss must have at least one example before it can be computed.')
        if self._reduce:
            return self._sum / self._num_steps
        else:
            return self._sum


class ActivationsBasedMetric(ABC, Metric):
    def __init__(self, indexes, output_transform=lambda x: x):
        super(ActivationsBasedMetric, self).__init__(output_transform)
        self.indexes = indexes
        self.activations = []

    def reset(self):
        self.activations = []

    def update(self, output):
        """
        Update the internals to be able to calculate the metric in the end
        :param output: state dictionary from the test/validate forward pass
        :return:
        """
        for index in self.indexes:
            output = output[index]

        batch_size = output.size()[0]
        features = output.cpu().data.numpy().reshape(batch_size, -1)
        self.activations.append(features)


