from os import path as osp

import numpy as np
import pandas as pd
from torchvision.transforms import transforms

from hibashi.data.aug import AugmentSample
from hibashi.data.datasets.datasets import Dataset
from hibashi.data.transformer import NormalizeSample
from hibashi.data.transformer import ToTensor, ToFloatTensor
from hibashi.utils.io import read_image


class FashionFinetune(Dataset):
    """
    Fashion Dataset Mother class
    """

    def __init__(self,
                 pd_df_rel_path: str,
                 images_rel_path: str = 'trimmed_images',
                 base_data_path: str = '/home/ubuntu/fashion-dataset',
                 subsample=None, aug_names=(), **kwargs):
        """
        :param pd_df_rel_path: relative path to the labels file, a pickled pandas dataframe
        :param images_rel_path: relative path to the images
        :param base_data_path: absolute path pointing to the root of the data
        :param subsample: (int) subsample the dataset based on input number
        :param aug_names: (n-tuple of str) with each entry a name for the augmentations
        :param kwargs:
        """
        super(FashionFinetune, self).__init__()

        self.images_path = osp.join(base_data_path, images_rel_path)
        self.data_frame = pd.read_pickle(osp.join(base_data_path, pd_df_rel_path))
        self.indexes = list(range(len(self.data_frame)))

        if subsample is not None:
            n_samples = subsample
            if isinstance(subsample, float):
                n_samples = int(len(self.data_frame) * subsample)
            self.indexes = list(np.random.choice(self.indexes, n_samples, replace=False))

        self.n_samples = len(self.indexes)

        self.transform = None
        self.aug_names = aug_names
        self.set_augmentations_and_transformations(aug_names)

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

    def set_augmentations_and_transformations(self, aug_names: tuple):
        """
        Set the augmentations and compose the transformation steps
        :param aug_names: the names of the augmentation classes you want to apply
        :return: None
        """
        self.aug_names = aug_names

        transform_steps = []
        if len(aug_names) > 0:
            transform_steps.append(AugmentSample(aug_names))

        transform_steps.append(ToTensor({'image': 'transpose_from_numpy', 'cls_idx': 'tensor'}))
        transform_steps.append(ToFloatTensor(['image']))
        transform_steps.append(NormalizeSample(('image',), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = transforms.Compose(transform_steps)

    def __getitem__(self, idx):
        """
        Get sample based on index
        :param idx: (int) index to retrieve
        :return: (dict) sample with keys: image, target (position map), ethnicity and gender
        """
        index = self.indexes[idx]
        row = self.data_frame.iloc[index]
        image_path = osp.join(self.images_path, row['image'])
        image = read_image(image_path)
        sample = {'image': image,
                  'article_type': row['articleType'],
                  'cls_idx': self.article_type_2_cls_idx[row['articleType']],
                  'idx': idx}
        sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

    @property
    def finite(self) -> bool:
        """
        :return: Is this dataset finite?
        """
        return True


class FashionFinetuneTrain(FashionFinetune):
    """
    Child class for the training part of the finetune dataset
    """
    def __init__(self, subsample=None, aug_names=(), **kwargs):
        super(FashionFinetuneTrain, self).__init__(
            pd_df_rel_path='finetune_train.df.pkl', subsample=subsample, aug_names=aug_names, **kwargs)


class FashionFinetuneVal(FashionFinetune):
    """
    Child class for the validation part of the finetune dataset
    """
    def __init__(self, subsample=None, aug_names=(), **kwargs):
        super(FashionFinetuneVal, self).__init__(
            pd_df_rel_path='finetune_val.df.pkl', subsample=subsample, aug_names=aug_names, **kwargs)


class FashionFinetuneTest(FashionFinetune):
    """
    Child class for the test part of the finetune dataset
    """
    def __init__(self, subsample=None, aug_names=(), **kwargs):
        super(FashionFinetuneTest, self).__init__(
            pd_df_rel_path='finetune_test.df.pkl', subsample=subsample, aug_names=aug_names, **kwargs)
