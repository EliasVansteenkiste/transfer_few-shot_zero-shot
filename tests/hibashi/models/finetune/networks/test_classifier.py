import torch

from hibashi.models.pretrain.networks.classification import ImageClassifier


class TestNetworks:

    def test_classifier(self):

        net = ImageClassifier()

        zeros = torch.zeros(2, 3, 128, 128)
        ones = torch.ones(2, 3, 128, 128)
        rand = torch.randn(2, 3, 128, 128)

        assert net(zeros).size() == (2, 20)
        assert net(ones).size() == (2, 20)
        assert net(rand).size() == (2, 20)


