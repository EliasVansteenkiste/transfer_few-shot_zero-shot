import torch.nn as nn

from hibashi.networks.network import Network
from hibashi.models.pretrain.networks.resnet import resnext50_32x4d


class ImageClassifier(Network):

    def __init__(self, num_classes=57, **kwargs):
        super(ImageClassifier, self).__init__(**kwargs)

        self.encoder = resnext50_32x4d(pretrained=True)
        self.fc = nn.Linear(self.encoder.n_features_before_classification, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)

