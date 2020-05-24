import torch.nn as nn
from torch.nn import init

from hibashi.networks.network import Network
from hibashi.models.pretrain.networks.resnet import resnext50_32x4d


class ImageClassifier(Network):

    def __init__(self, num_classes=20, pretrained=True, **kwargs):
        super(ImageClassifier, self).__init__(**kwargs)

        self.encoder = resnext50_32x4d(pretrained=pretrained)
        self.fc = nn.Linear(self.encoder.n_features_before_classification, num_classes)

    def init_linear_classifier(self):
        init.orthogonal_(self.fc.weight)
        init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)

