from os import path as osp

import numpy as np
import pandas as pd
from torchvision.transforms import transforms

from hibashi.data.aug import AugmentSample
from hibashi.data.datasets.datasets import Dataset
from hibashi.data.transformer import NormalizeSample
from hibashi.data.transformer import ToTensor, ToFloatTensor
from hibashi.utils.io import read_image


class Fashion(Dataset):
    """
    Fashion Dataset Mother class
    """

    def __init__(self,
                 pd_df_rel_path: str,
                 images_rel_path: str = 'trimmed_images',
                 base_data_path: str = '/Users/elias/Downloads/fashion-dataset',
                 subsample=None, aug_names=(), **kwargs):
        """
        :param pd_df_rel_path: relative path to the labels file, a pickled pandas dataframe
        :param images_rel_path: relative path to the images
        :param base_data_path: absolute path pointing to the root of the data
        :param subsample: (int) subsample the dataset based on input number
        :param aug_names: (n-tuple of str) with each entry a name for the augmentations
        :param kwargs:
        """
        super(Fashion, self).__init__()

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

        self.article_type_2_cls_idx = {"Jeans": 19,
                                       "Perfume and Body Mist": 18,
                                       "Formal Shoes": 17,
                                       "Socks": 16,
                                       "Backpacks": 15,
                                       "Belts": 14,
                                       "Briefs": 13,
                                       "Sandals": 12,
                                       "Flip Flops": 11,
                                       "Wallets": 10,
                                       "Sunglasses": 9,
                                       "Heels": 8,
                                       "Handbags": 7,
                                       "Tops": 6,
                                       "Kurtas": 5,
                                       "Sports Shoes": 4,
                                       "Watches": 3,
                                       "Casual Shoes": 2,
                                       "Shirts": 1,
                                       "Tshirts": 0}

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

        transform_steps.append(NormalizeSample({'image': (0, 255)}))
        transform_steps.append(ToTensor({'image': 'transpose_from_numpy',
                                         'cls_idx': 'tensor'}))
        transform_steps.append(ToFloatTensor(['image']))

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


class FashionPretrainTrain(Fashion):
    """
    Child class for the training part of the pretrain dataset with the 20 largest article_types
    """

    def __init__(self, subsample=None, aug_names=(), **kwargs):
        super(FashionPretrainTrain, self).__init__(
            pd_df_rel_path='pretrain_train.df.pkl', subsample=subsample, aug_names=aug_names, **kwargs)


class FashionPretrainVal(Fashion):
    """
    Child class for the validation part of the pretrain dataset with the 20 largest article_types
    """

    def __init__(self, subsample=None, aug_names=(), **kwargs):
        super(FashionPretrainVal, self).__init__(
            pd_df_rel_path='pretrain_val.df.pkl', subsample=subsample, aug_names=aug_names, **kwargs)


class FashionPretrainTest(Fashion):
    """
    Child class for the test part of the pretrain dataset with the 20 largest article_types
    """

    def __init__(self, subsample=None, aug_names=(), **kwargs):
        super(FashionPretrainTest, self).__init__(
            pd_df_rel_path='pretrain_test.df.pkl', subsample=subsample, aug_names=aug_names, **kwargs)
