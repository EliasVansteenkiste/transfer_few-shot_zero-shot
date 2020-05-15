import numpy as np
import torch

from hibashi.models.pretrain.losses import ConfusionMatrix, FocalLoss


class TestConfusionMatrix:
    """
    Some basic sanity tests
    """
    def test_basic(self):

        cls_idx_2_article_type = {19: "Jeans",
                                  18: "Perfume and Body Mist",
                                  17: "Formal Shoes",
                                  16: "Socks",
                                  15: "Backpacks",
                                  14: "Belts",
                                  13: "Briefs",
                                  12: "Sandals",
                                  11: "Flip Flops",
                                  10: "Wallets",
                                  9: "Sunglasses",
                                  8: "Heels",
                                  7: "Handbags",
                                  6: "Tops",
                                  5: "Kurtas",
                                  4: "Sports Shoes",
                                  3: "Watches",
                                  2: "Casual Shoes",
                                  1: "Shirts",
                                  0: "Tshirts"}
        cm = ConfusionMatrix(cls_idx_2_article_type)

        pred = torch.Tensor([[0., 0., .01, .99],
                             [0., 0., 0., 1.],
                             [0., 0., 20., 80.],
                             [0., 0., 100., 1.]])

        target = torch.Tensor([3, 3, 0, 0])

        result = cm(pred, target)

        expected = np.zeros((20, 20))
        expected[3, 3] += 2
        expected[0, 3] += 1
        expected[0, 2] += 1

        assert (result == expected).all()


class TestFocalLoss:
    """
    Some basic sanity tests
    """
    def test_basic(self):

        floss = FocalLoss()

        pred = torch.Tensor([[0., 0., .01, .99],
                             [0., 0., 0., 1.],
                             [0., 0., 20., 80.],
                             [0., 0., 100., 1.]])

        close_target = torch.Tensor([3, 3, 3, 2]).long()
        far_target = torch.Tensor([0, 1, 0, 1]).long()

        assert floss(pred, close_target) < 0.1
        assert floss(pred, far_target) > 2




