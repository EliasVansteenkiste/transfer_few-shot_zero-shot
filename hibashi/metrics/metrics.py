from abc import ABC

import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from torch.nn import functional as tnnf


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

    def __str__(self):
        return self.__class__.__name__+'-'+self._loss_name


class Fbeta(Metric):
    """
    Calculate F1 metric over the whole validation set
    """

    def __init__(self, cls_idx: int, num_classes: int, beta: int = 1, epsilon=1e-8):
        super(Fbeta, self).__init__()

        self.beta = beta
        self.num_classes = num_classes
        self.cls_idx = cls_idx
        self.epsilon = epsilon

        self.tp = 0
        self.fp = 0
        self.fn = 0

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, output):
        """
        Calculates the elements of a confusion matrix according to the predicted and target values.
        :param pred: network prediction
        :param target: correspondent target
        :return: confusion matrix array values
        """
        pred = output['out_pred_cls']
        target = output['in_cls_targets']

        target = tnnf.one_hot(target, num_classes=self.num_classes)
        pred = tnnf.one_hot(pred.argmax(1), num_classes=self.num_classes)

        target = target[:, self.cls_idx]
        pred = pred[:, self.cls_idx]

        self.tp += torch.sum(pred*target)
        self.fp += torch.sum((1-pred)*target)
        self.fn += torch.sum(pred*(1-target))

    def compute(self):
        return (1 + self.beta**2) * self.tp /\
               ((1 + self.beta**2) * self.tp + self.beta**2 * self.fn + self.fp + self.epsilon)

    def __str__(self):
        return f'F{self.beta}_{self.cls_idx}th_class_of_{self.num_classes}_classes'


class TopKAccuracy(Metric):
    """
    Calculate Top K Accuracy metric over the whole validation set
    """

    def __init__(self, num_classes: int, top_k: int = 1):
        super(TopKAccuracy, self).__init__()

        self.num_classes = num_classes
        self.top_k = top_k

        self.tp = 0
        self.total = 0

    def reset(self):
        self.tp = 0
        self.total = 0

    def update(self, output):
        """
        Calculates the elements of a confusion matrix according to the predicted and target values.
        :param pred: network prediction
        :param target: correspondent target
        :return: confusion matrix array values
        """
        pred = output['out_pred_cls']
        target = output['in_cls_targets']

        target = tnnf.one_hot(target, num_classes=self.num_classes)
        pred = tnnf.one_hot(torch.topk(pred, self.top_k)[1], num_classes=self.num_classes)
        self.tp += torch.sum(pred*target[:, None, :], dim=(0, 1))
        self.total += torch.sum(target, dim=0)

    def compute(self):
        return (self.tp.float() / self.total.float()).mean()

    def compute_per_idx(self, cls_idx):
        return 1. * self.tp[cls_idx].float() / self.total[cls_idx].float()

    def __str__(self):
        return f'Top{self.top_k}_{self.num_classes}_classes'


class AverageAccuracy(Metric):
    """
    Calculate Average Accuracy with equal weight for every category in your test or validation set
    """

    def __init__(self, num_classes: int):
        super(AverageAccuracy, self).__init__()

        self.num_classes = num_classes
        self.incidences = 0
        self.true_positives = 0

    def reset(self):
        self.incidences = 0
        self.true_positives = 0

    def update(self, output):
        """
        Calculates the elements of a confusion matrix according to the predicted and target values.
        :param pred: network prediction
        :param target: correspondent target
        :return: confusion matrix array values
        """
        pred = output['out_pred_cls']
        cls_target = output['in_cls_targets']

        target = tnnf.one_hot(cls_target, num_classes=self.num_classes)
        pred = tnnf.one_hot(pred.argmax(1), num_classes=self.num_classes)

        self.incidences += torch.sum(target, dim=0)
        self.true_positives += torch.sum(pred * target, dim=0)

    def compute(self):
        accuracies = []
        incidences = self.incidences.float()
        true_positives = self.true_positives.float()
        for incidence, true_positives in zip(incidences, true_positives):
            if incidence > 0:
                accuracies.append(true_positives/incidence)
        return torch.mean(torch.stack(accuracies))

    def __str__(self):
        return self.__class__.__name__ + f'-{self.num_classes}_classes'


class AverageTop1ErrorRate(Metric):
    """
    Calculate Average Top 1 Error Rate, every category has an equal weight
    """

    def __init__(self, num_classes):
        super(AverageTop1ErrorRate, self).__init__()

        self.num_classes = num_classes
        self.incidences = 0
        self.true_positives = 0

    def reset(self):
        self.incidences = 0
        self.true_positives = 0

    def update(self, output):
        """
        Calculates the elements of a confusion matrix according to the predicted and target values.
        :param pred: network prediction
        :param target: correspondent target
        :return: confusion matrix array values
        """
        pred = output['out_pred_cls']
        cls_target = output['in_cls_targets']

        target = tnnf.one_hot(cls_target, num_classes=self.num_classes)
        pred = tnnf.one_hot(pred.argmax(1), num_classes=self.num_classes)

        self.incidences += torch.sum(target, dim=0)
        self.true_positives += torch.sum(pred * target, dim=0)

    def compute(self):
        error_rates = []
        incidences = self.incidences.float()
        true_positives = self.true_positives.float()
        for incidence, true_positives in zip(incidences, true_positives):
            if incidence > 0:
                error_rates.append(1-true_positives/incidence)
        return torch.mean(torch.stack(error_rates))

    def __str__(self):
        return self.__class__.__name__ + f'-{self.num_classes}_classes'


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


