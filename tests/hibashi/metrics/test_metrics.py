import numpy as np
import torch

from hibashi.metrics.metrics import Fbeta
from hibashi.models.pretrain.losses import ConfusionMatrix


class TestFBeta:
    """
    Some basic sanity tests
    """
    def test_basic(self):

        f1_0 = Fbeta(cls_idx=0, num_classes=3, beta=1)
        f1_1 = Fbeta(cls_idx=1, num_classes=3, beta=1)
        f1_2 = Fbeta(cls_idx=2, num_classes=3, beta=1)


        pred = torch.Tensor([[1, 0, 0.],
                             [1, 0, 0.],
                             [0., 1, 0],
                             [0., 1, 0],
                             [0., 0, 1],
                             [0., 0, 1]])

        target = torch.Tensor([0, 1, 1, 2, 2, 0]).long()

        output = {'out_pred_cls': pred, 'in_cls_targets': target}

        f1_0.update(output)
        f1_1.update(output)
        f1_2.update(output)

        result0 = f1_0.compute()
        result1 = f1_1.compute()
        result2 = f1_2.compute()

        assert result0 == 0.5
        assert result1 == 0.5
        assert result2 == 0.5
        