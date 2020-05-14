import torch.nn as nn
from torch.optim import Adam

from hibashi.framework.factory import factory
from hibashi.losses.loss import Loss
from hibashi.models import Model
from hibashi.models.pretrain.losses import CE, Accuracy, ConfusionMatrix
from hibashi.models.pretrain.networks.classification import ImageClassifier
from hibashi.models.pretrain.utils import plot_confusion_matrix_validation


class PreTrain(Model):

    def __init__(self, gpu_ids, is_train: bool, **kwargs):
        """

        :param gpu_ids: list with gpu ids
        :param is_train: flag for model being in training modus
        :param kwargs:
        """
        Model.__init__(self, gpu_ids, is_train=is_train)

        self.is_train = is_train

        self.net_classifier = self.define_net()

        self.in_imgs = None
        self.in_real_paths = None
        self.in_cls_targets = None
        self.out_pred_cls = None

        self.loss_cls = None
        self.loss_accuracy = None
        self.non_scalar_metric_cm = None

        self.cls_idx_2_article_type = {19: "Jeans",
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

        self.criterion_cls = CE()
        self.criterion_acc = Accuracy()
        self.criterion_cm = ConfusionMatrix(self.cls_idx_2_article_type)

        if self.is_train:
            self.optimizer = Adam(self.net_classifier.parameters(), lr=0.0002, betas=(0., 0.9), weight_decay=0, eps=1e-8)

            # Constant learning rate for now
            self.schedulers = []

    @property
    def name_main_metric(self):
        return "loss_accuracy"

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
        self.non_scalar_metric_cm = self.criterion_cm(self.out_pred_cls, self.in_cls_targets)

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

    def get_validation_figures(self, state: dict) -> dict:
        """
        Creates a dictionary containing all the figures for validation.
        :param state: (engine object) that contains the current state items
        :return: out_dict: with key/value : name/matplotlib figure objects
        """
        out_dict = {}
        for name, metric in state.metrics.items():
            if name == 'non_scalar_metric_cm':
                fig = plot_confusion_matrix_validation(metric, self.criterion_cm.labels)
                out_dict['cm'] = fig
            else:
                continue
        return out_dict

    @property
    def visuals(self) -> dict:
        """
        Collect and construct the visuals
        :return: dictionary with
        """
        visuals = {}
        return visuals
