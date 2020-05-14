import numpy as np
import torch
from torch.nn import functional as tnnf
from sklearn.metrics import confusion_matrix

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


class ConfusionMatrix(Loss):
    def __init__(self, cls_idx_2_label):
        super(ConfusionMatrix, self).__init__()
        self.cls_idx_2_label = cls_idx_2_label
        self.labels = [label for cls_idx, label in sorted(cls_idx_2_label.items(), key=lambda x: x[0])]

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
        """
        Calculates the elements of a confusion matrix according to the predicted and target values.
        :param pred: network prediction
        :param target: correspondent target
        :return: confusion matrix array values
        """
        target = target.cpu().data.numpy()
        pred = tnnf.softmax(pred, dim=1).cpu().data.numpy()

        predictions = []
        targets = []
        for batch_idx in range(len(target)):
            predictions.append(self.cls_idx_2_label[np.argwhere(np.max(pred[batch_idx]) == pred[batch_idx])[0, 0]])
            targets.append(self.cls_idx_2_label[target[batch_idx]])

        cm = confusion_matrix(targets, predictions, labels=self.labels)
        return cm

