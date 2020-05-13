import torch.nn as nn
from torch.optim import Adam

from hibashi.framework.factory import factory
from hibashi.losses.loss import Loss
from hibashi.models import Model
from hibashi.models.pretrain.losses import CE, Accuracy
from hibashi.models.pretrain.networks.classification import ImageClassifier


class PreTrain(Model):

    def __init__(self, gpu_ids, is_train: bool, **kwargs):
        """

        :param gpu_ids: list with gpu ids
        :param is_train: flag for model being in training modus
        :param kwargs:
        """
        Model.__init__(self, gpu_ids, is_train=is_train)

        self.criterion_cls = CE()
        self.criterion_acc = Accuracy()

        self.is_train = is_train

        self.net_classifier = self.define_net()

        self.in_imgs = None
        self.in_real_paths = None
        self.in_cls_targets = None
        self.out_pred_cls = None

        self.loss_cls = None
        self.loss_accuracy = None

        if self.is_train:
            self.optimizer = Adam(self.net_classifier.parameters(), lr=0.0002, betas=(0., 0.9), weight_decay=0, eps=1e-8)

            # Constant learning rate for now
            self.schedulers = []

    @property
    def name_main_metric(self):
        return "loss_cls"

    def main_metric(self, metrics: dict):
        """
        Calculate the main metric based on all previously calculated metrics
        :param metrics:
        :return:
        """
        return metrics[self.name_main_metric]

    def define_net(self):
        net = ImageClassifier()
        if self.device.type == 'cuda':
            net.to(self.device)
        return net

    def set_input(self, batch):
        """
        Unpack input data from the data loader and perform necessary pre-processing steps.
        :param  batch: a dictionary that contains the data itself and its metadata information.
        """
        images = batch['image']
        cls_targets = batch['cls_idx'].squeeze()

        self.in_cls_targets = cls_targets.to(self.device, non_blocking=True)
        self.in_imgs = images.to(self.device, non_blocking=True)

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.out_pred_cls = self.net_classifier.forward(self.in_imgs)

    def calculate_metrics(self):
        """Calculate metrics which are used during evaluation. This should also include all the losses"""
        self.loss_cls = self.criterion_cls(self.out_pred_cls, self.in_cls_targets)
        self.loss_accuracy = self.criterion_acc(self.out_pred_cls, self.in_cls_targets)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.calculate_metrics()
        self.loss_cls.backward()

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    @property
    def visuals(self) -> dict:
        """
        Collect and construct the visuals
        :return: dictionary with
        """
        visuals = {}
        return visuals
