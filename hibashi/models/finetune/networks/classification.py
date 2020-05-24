import torch
import torch.nn as nn
from torch.nn import init

from hibashi.networks.network import Network
from hibashi.models.finetune.networks.resnet import resnext50_32x4d


class ImageClassifier(Network):

    def __init__(self, num_classes=57, pretrained=False, **kwargs):
        super(ImageClassifier, self).__init__(**kwargs)

        self.encoder = resnext50_32x4d(pretrained=pretrained)
        self.fc = nn.Linear(self.encoder.n_features_before_classification, num_classes)

    def load_weights_into_encoder(self, path: str, device):
        weights = torch.load(path, map_location=device)
        del weights['fc.weight']
        del weights['fc.bias']
        results = self.load_state_dict(weights, strict=False)
        assert set(results.missing_keys) == {'fc.weight', 'fc.bias'}

    def init_linear_classifier(self):
        init.orthogonal_(self.fc.weight)
        init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x)


class ImageClassifierV2(Network):

    def __init__(self, num_classes=57, pretrained=False, **kwargs):
        super(ImageClassifierV2, self).__init__(**kwargs)

        self.encoder = resnext50_32x4d(pretrained=pretrained)
        self.lrelu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(p=0.5)
        self.fc0 = nn.Linear(self.encoder.n_features_before_classification, 1024)
        self.fc1 = nn.Linear(1024, num_classes)

    def load_weights_into_encoder(self, path: str, device):
        weights = torch.load(path, map_location=device)
        del weights['fc.weight']
        del weights['fc.bias']
        results = self.load_state_dict(weights, strict=False)
        assert set(results.missing_keys) == {'fc0.weight', 'fc0.bias', 'fc1.weight', 'fc1.bias'}

    def init_linear_classifier(self):
        init.orthogonal_(self.fc0.weight)
        init.zeros_(self.fc0.bias)
        init.orthogonal_(self.fc1.weight)
        init.zeros_(self.fc1.bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.drop(x)
        x = self.fc0(x)
        x = self.lrelu(x)
        x = self.fc1(x)
        return x

