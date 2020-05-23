import torch
from ignite.contrib.handlers import PiecewiseLinear
from torch.optim import Adam

from hibashi.models import Model
from hibashi.models.finetune.losses import CE, Accuracy, ConfusionMatrix, FocalLoss
from hibashi.models.finetune.networks.classification import ImageClassifier
from hibashi.models.finetune.utils import plot_confusion_matrix_validation


class Finetune(Model):

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
        self.loss_focal = None
        self.loss_error_rate = None
        self.loss_cls_weighted = None

        self.non_scalar_metric_cm = None

        self.article_type_2_cls_idx = {"Accessory Gift Set": 0,
                                       "Bangle": 1,
                                       "Bra": 2,
                                       "Bracelet": 3,
                                       "Camisoles": 4,
                                       "Capris": 5,
                                       "Caps": 6,
                                       "Churidar": 7,
                                       "Clutches": 8,
                                       "Cufflinks": 9,
                                       "Dresses": 10,
                                       "Duffel Bag": 11,
                                       "Dupatta": 12,
                                       "Earrings": 13,
                                       "Flats": 14,
                                       "Footballs": 15,
                                       "Free Gifts": 16,
                                       "Gloves": 17,
                                       "Headband": 18,
                                       "Jackets": 19,
                                       "Jewellery Set": 20,
                                       "Jumpsuit": 21,
                                       "Kurta Sets": 22,
                                       "Kurtis": 23,
                                       "Laptop Bag": 24,
                                       "Leggings": 25,
                                       "Lounge Pants": 26,
                                       "Lounge Shorts": 27,
                                       "Messenger Bag": 28,
                                       "Mobile Pouch": 29,
                                       "Mufflers": 30,
                                       "Necklace and Chains": 31,
                                       "Night suits": 32,
                                       "Nightdress": 33,
                                       "Pendant": 34,
                                       "Rain Jacket": 35,
                                       "Ring": 36,
                                       "Rucksacks": 37,
                                       "Scarves": 38,
                                       "Shoe Accessories": 39,
                                       "Shorts": 40,
                                       "Skirts": 41,
                                       "Sports Sandals": 42,
                                       "Stockings": 43,
                                       "Stoles": 44,
                                       "Sweaters": 45,
                                       "Sweatshirts": 46,
                                       "Swimwear": 47,
                                       "Ties": 48,
                                       "Track Pants": 49,
                                       "Tracksuits": 50,
                                       "Travel Accessory": 51,
                                       "Trousers": 52,
                                       "Tunics": 53,
                                       "Waist Pouch": 54,
                                       "Waistcoat": 55,
                                       "Wristbands": 56}

        self.cls_idx_2_article_type = {}
        for article_type, cls_idx in self.article_type_2_cls_idx.items():
            self.cls_idx_2_article_type[cls_idx] = article_type

        n_samples_in_train = {'Dresses': 260, 'Flats': 239, 'Shorts': 235, 'Earrings': 235, 'Trousers': 188,
                              'Clutches': 153, 'Tunics': 129, 'Ties': 126, 'Caps': 106, 'Bra': 104, 'Kurtis': 102,
                              'Track Pants': 92, 'Leggings': 90, 'Capris': 86, 'Necklace and Chains': 84, 'Dupatta': 76,
                              'Pendant': 66, 'Stoles': 56, 'Free Gifts': 55, 'Cufflinks': 53, 'Skirts': 53,
                              'Scarves': 51, 'Night suits': 50, 'Jackets': 49, 'Bangle': 43, 'Kurta Sets': 43,
                              'Ring': 42, 'Laptop Bag': 33, 'Bracelet': 32, 'Jewellery Set': 32, 'Duffel Bag': 29,
                              'Lounge Pants': 29, 'Sports Sandals': 26, 'Sweatshirts': 20, 'Mobile Pouch': 20,
                              'Nightdress': 20, 'Churidar': 18, 'Lounge Shorts': 18, 'Sweaters': 15,
                              'Messenger Bag': 14, 'Stockings': 11, 'Rain Jacket': 11, 'Gloves': 10, 'Jumpsuit': 10,
                              'Accessory Gift Set': 10, 'Swimwear': 6, 'Tracksuits': 6, 'Mufflers': 6, 'Waistcoat': 6,
                              'Travel Accessory': 4, 'Footballs': 4, 'Camisoles': 4, 'Waist Pouch': 3, 'Headband': 3,
                              'Shoe Accessories': 3, 'Rucksacks': 2, 'Wristbands': 2}

        print(len(n_samples_in_train))

        ce_weights = [1000 / n_samples_in_train[self.cls_idx_2_article_type[i]] for i in range(57)]
        self.ce_weights = torch.Tensor(ce_weights).to(self.device, non_blocking=True)

        self.criterion_cls = CE()
        self.criterion_acc = Accuracy()
        self.criterion_focal = FocalLoss()
        self.criterion_cm = ConfusionMatrix(self.cls_idx_2_article_type)

        if self.is_train:
            self.optimizer = Adam(self.net_classifier.parameters(), lr=0.0002, betas=(0., 0.9), weight_decay=0,
                                  eps=1e-8)
            self.schedulers = self.assign_schedulers()

    def assign_schedulers(self):
        """
        Assigns the step schedulers for each network parameters group
        :return: (list) with parameters group step schedulers for ignite
        """
        schedulers = []

        milestones = ((0, 2e-4),
                      (8499, 2e-4),
                      (8500, 1e-4),
                      (16999, 1e-4),
                      (17000, 5e-5),
                      (25999, 5e-5),
                      (26000, 2.5e-5))

        multi_lr = PiecewiseLinear(self.optimizer, "lr", milestones_values=milestones)
        schedulers.append(multi_lr)
        return schedulers

    @property
    def name_main_metric(self):
        return "AverageTop1ErrorRateFinetune"

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

    def load_weights(self, path: str):
        weights = torch.load(path, map_location=self.device)
        self.net_classifier.load_state_dict(weights)

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
        self.loss_cls_weighted = self.criterion_cls(self.out_pred_cls, self.in_cls_targets)
        self.loss_focal = self.criterion_focal(self.out_pred_cls, self.in_cls_targets)
        self.loss_accuracy = self.criterion_acc(self.out_pred_cls, self.in_cls_targets)
        self.loss_error_rate = 1 - self.loss_accuracy
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
